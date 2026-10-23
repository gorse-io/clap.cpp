#include "clap-internal.h"
#include "feature-test.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

static void equal(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) throw std::runtime_error("dimension mismatch");
    for (size_t i=0;i<a.size();++i)
        if (!std::isfinite(a[i]) || std::abs(a[i]-b[i])>1e-6f)
            throw std::runtime_error("bundled pipeline differs from dual models");
}
template<class F> static void rejects(F fn) {
    try { fn(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid input accepted");
}
int main(int argc, char **argv) {
    try {
        if (argc!=4) throw std::runtime_error("bundled, feature-only, MERT models required");
        clap::Model bundled(argv[1],2), target(argv[2],2), mert(argv[3],2);
        if (bundled.audio_sample_rate()!=24000) throw std::runtime_error("bundled sample rate");
        if (target.audio_sample_rate()!=24000) throw std::runtime_error("mandatory frontend sample rate");
        std::vector<float> pcm(48000);
        for (size_t i=0;i<pcm.size();++i) pcm[i]=0.2f*std::sin(i*0.071f)+0.1f;
        const auto resampled=clap::resample_audio({pcm,48000},24000).samples;
        const auto features=mert.clamp3_features(resampled,24000);
        const auto expected=clap::detail::FeatureTestAccess::encode(argv[2],features.values,features.frames);
        equal(bundled.audio(pcm,48000),expected);
        equal(bundled.audio(resampled,24000),expected);
        equal(clap::detail::FeatureTestAccess::encode(argv[1],features.values,features.frames),expected);
        equal(bundled.text("ab c"),target.text("ab c"));
        rejects([&]{ bundled.audio({},24000); });
        rejects([&]{ bundled.audio(std::vector<float>(23999),24000); });
        rejects([&]{ bundled.audio(pcm,0); });
        rejects([&]{ bundled.audio(pcm,24000,1); });
        rejects([&]{ bundled.audio_frames(pcm,24000); });
        rejects([&]{ bundled.clamp3_features(pcm,24000); });
        pcm[0]=std::numeric_limits<float>::quiet_NaN();
        rejects([&]{ bundled.audio(pcm,24000); });
        std::cout<<"Bundled CLaMP3 native waveform/resampling/features/text/invalid-input passed\n";
        return 0;
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
