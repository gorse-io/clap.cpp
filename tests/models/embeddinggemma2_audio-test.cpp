#include "models/embeddinggemma2-audio.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace clap::detail {
struct GemmaAudioTestAccess {
    static Matrix features(const std::vector<float> &pcm) { return GemmaAudioEncoder::log_mel(pcm); }
    static Matrix encode(const GemmaAudioEncoder &encoder, const Matrix &features) { return encoder.encode_features(features); }
};
}
using namespace clap::detail;
static void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static float compare(const Matrix &actual, const nlohmann::json &expected, float tolerance) {
    require(actual.rows == int(expected.size()), "row count mismatch");
    float maximum = 0;
    for (int r=0;r<actual.rows;++r) {
        require(actual.cols == int(expected[r].size()), "column count mismatch");
        for (int c=0;c<actual.cols;++c) {
            require(std::isfinite(actual[r][c]), "non-finite output");
            maximum=std::max(maximum,std::abs(actual[r][c]-expected[r][c].get<float>()));
        }
    }
    std::cout << "max_abs_error=" << maximum << " tolerance=" << tolerance << '\n';
    require(maximum < tolerance, "independent audio oracle mismatch");
    return maximum;
}
int main(int argc,char **argv) {
    try {
        require(argc>=3,"usage: audio-test MODEL FIXTURE [rejection-substring]");
        ModelLoader loader; loader.load(argv[1]);
        if(argc==4 && std::string(argv[3])=="--dump") {
            GemmaAudioEncoder encoder(loader,2);
            std::ifstream stream(argv[2]); nlohmann::json input; stream>>input;
            auto features=GemmaAudioTestAccess::features(input.at("pcm").get<std::vector<float>>());
            auto encoded=GemmaAudioTestAccess::encode(encoder,features);
            auto rows=[](const Matrix &m) {
                auto result=nlohmann::json::array();
                for(int r=0;r<m.rows;++r) result.push_back(std::vector<float>(m[r],m[r]+m.cols));
                return result;
            };
            std::cout << nlohmann::json{{"features",rows(features)},{"encoded",rows(encoded)}}.dump() << '\n';
            return 0;
        }
        if(argc==4) {
            try { GemmaAudioEncoder encoder(loader,2); }
            catch(const std::exception &e) {
                require(std::string(e.what()).find(argv[3])!=std::string::npos,e.what()); return 0;
            }
            throw std::runtime_error("invalid audio model accepted");
        }
        GemmaAudioEncoder encoder(loader,2);
        // Official Gemma4 adds the mel floor; max(mel,floor) is not equivalent.
        const auto quiet = GemmaAudioTestAccess::features(std::vector<float>(321, 1e-6f));
        // Bin zero has no interior FFT frequency at this mel resolution.
        require(quiet[0][1] > std::log(0.001f) + 1e-6f, "mel floor must be additive");
        std::ifstream stream(argv[2]); nlohmann::json fixture; stream>>fixture;
        for(const auto &test:fixture.at("frontend")) {
            auto pcm=test.at("pcm").get<std::vector<float>>();
            auto features=GemmaAudioTestAccess::features(pcm);
            compare(features,test.at("features"),2e-5f);
            compare(encoder.encode(pcm),test.at("encoded"),fixture.value("tolerance",2e-5f));
        }
        for(const auto &test:fixture.at("tower")) {
            const auto &f=test.at("features"); Matrix features(int(f.size()),128);
            for(int r=0;r<features.rows;++r) for(int c=0;c<128;++c) features[r][c]=f[r][c].get<float>();
            auto actual=GemmaAudioTestAccess::encode(encoder,features);
            compare(actual,test.at("encoded"),fixture.value("tolerance",2e-5f));
            require(actual.data==GemmaAudioTestAccess::encode(encoder,features).data,"nondeterministic audio");
        }
        // Frame-end masks and two ceil/2 subsamplers, not the stale 280 cap.
        if (fixture.value("boundaries", false)) {
            for (int samples : {161, 320, 16000, 16001, 480000, 480100}) {
                std::vector<float> pcm(samples, 0.0f);
                const int frames = (std::min(samples, 480000) - 161) / 160 + 1;
                require(GemmaAudioTestAccess::features(pcm).rows == frames, "wrong boundary frame count");
                require(encoder.encode(pcm).rows == (frames + 3) / 4, "wrong boundary soft-token count");
            }
            std::vector<float> long_pcm(480100, 0.0f);
            std::fill(long_pcm.begin() + 480000, long_pcm.end(), 0.9f);
            const auto truncated = encoder.encode(long_pcm);
            require(truncated.rows == 750, "30 seconds must produce 750 soft tokens");
            long_pcm.resize(480000);
            require(truncated.data == encoder.encode(long_pcm).data, "first-30-second truncation mismatch");
        }
        for(const auto &pcm: {std::vector<float>{},std::vector<float>(160,0),
                std::vector<float>(170,std::numeric_limits<float>::quiet_NaN()),
                std::vector<float>(170,std::numeric_limits<float>::infinity())}) {
            bool rejected=false; try { encoder.encode(pcm); }catch(const std::exception&){rejected=true;}
            require(rejected,"invalid PCM accepted");
        }
        return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
