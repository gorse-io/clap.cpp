#include "clap.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }
void compare(const std::vector<float> &a, const std::vector<float> &b, float tolerance, const std::string &name) {
    require(a.size() == b.size(), name + " size"); float error = 0;
    for (size_t i=0;i<a.size();++i) { require(std::isfinite(a[i]) && std::isfinite(b[i]), name + " finite"); error=std::max(error,std::abs(a[i]-b[i])); }
    std::cout << name << " max_abs_error=" << error << std::endl;
    require(error<=tolerance, name + " parity");
}
int main(int argc, char **argv) {
    try {
        for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            for (bool reference : {false, true}) {
                bool rejected = false;
                try { compare(reference ? std::vector<float>{0, 0} : std::vector<float>{0, bad},
                              reference ? std::vector<float>{0, bad} : std::vector<float>{0, 0}, 0, "nonfinite regression"); }
                catch (const std::exception &) { rejected = true; }
                require(rejected, "comparison must reject nonfinite native AND reference components");
            }
        }
        require(argc == 3, "usage: muq-mulan-model-test MODEL FIXTURE");
        nlohmann::json f; std::ifstream input(argv[2]); input >> f;
        clap::Model m(argv[1], 2);
        require(m.projection_dimension() == f.at("dimension"), "joint width");
        require(m.audio_sample_rate() == 24000, "rate");
        const float tol = f.at("tolerance");
        for (const auto &c : f.at("texts")) {
            const auto ids = c.at("ids").get<std::vector<int>>();
            require(m.tokenize(c.at("text"), 10000) == ids, "token parity");
            compare(m.text(ids), c.at("expected"), tol, "IDs: " + c.at("text").get<std::string>());
            compare(m.text(c.at("text").get<std::string>()), c.at("expected"), tol, "UTF8");
        }
        for (const auto &c : f.at("audio"))
            compare(m.audio(c.at("wave"),24000), c.at("expected"), tol, c.at("name"));
        auto reject = [&](auto fn, const std::string &name) { bool caught=false; try { fn(); } catch(const std::exception &) {caught=true;} require(caught,"must reject " + name); };
        const auto wave = f.at("audio").back().at("wave").get<std::vector<float>>();
        compare(m.audio(wave,48000), m.audio(clap::resample_audio({wave,48000},24000).samples,24000),0,"automatic native resampling");
        reject([&]{m.audio_frames(wave,24000);},"frames");
        reject([&]{m.clamp3_features(wave,24000);},"clamp3");
        reject([&]{m.audio(wave,24000,1);},"crop");
        reject([&]{m.audio(wave,0);},"rate");
        reject([&]{m.audio({},24000);},"empty");
        reject([&]{m.audio(std::vector<float>(512),24000);},"short");
        auto bad=wave; bad[0]=std::numeric_limits<float>::quiet_NaN();
        reject([&]{m.audio(bad,24000);},"nonfinite");
        reject([&]{m.text(std::vector<int>{});},"empty IDs");
        reject([&]{m.text(std::vector<int>{-1});},"invalid ID");
        reject([&]{m.text(std::vector<int>(600,0));},"over context IDs");
        reject([&]{m.text(std::string(3000,'a'));},"over context UTF8");
        std::cout << "MuLan native contracts passed" << std::endl;
    } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl; return 1;}
}
