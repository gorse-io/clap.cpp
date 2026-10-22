#include "feature-test.h"
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc!=3) throw std::runtime_error("MODEL.gguf FEATURES.json required");
        std::ifstream input(argv[2]); nlohmann::json data; input>>data;
        std::cout<<nlohmann::json(clap::detail::FeatureTestAccess::encode(argv[1],
            data.at("values").get<std::vector<float>>(),data.at("frames").get<int>())).dump()<<'\n';
        return 0;
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
