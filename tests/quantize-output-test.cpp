#include "quantize-output.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
std::filesystem::path output_directory;

std::string read(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

class QuantizeOutputTest : public testing::Test {
protected:
    std::filesystem::path destination;

    void SetUp() override {
        std::filesystem::remove_all(output_directory);
        std::filesystem::create_directories(output_directory);
        destination = output_directory / "output.gguf";
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove_all(output_directory, error);
        EXPECT_FALSE(error) << "output cleanup failed: " << error.message();
    }

    void ExpectOnlyDestinationRemains() {
        EXPECT_EQ(1, std::distance(std::filesystem::directory_iterator(output_directory),
                                  std::filesystem::directory_iterator())) << "temporary output leaked";
    }
};

TEST_F(QuantizeOutputTest, RejectsConcurrentDestinationWithoutClobberingIt) {
    {
        clap::detail::QuantizeOutput output(destination);
        output.write("quantized", 9);
        output.close();
        // Simulate a competitor winning immediately before atomic publication.
        std::ofstream(destination, std::ios::binary) << "competitor";
        EXPECT_THROW(output.publish(), std::exception);
        EXPECT_EQ("competitor", read(destination));
    }
    ExpectOnlyDestinationRemains();
}

TEST_F(QuantizeOutputTest, PublishesContentAndCleansTemporaryOutput) {
    {
        clap::detail::QuantizeOutput output(destination);
        output.write("quantized", 9);
        output.close();
        output.publish();
    }
    EXPECT_EQ("quantized", read(destination));
    ExpectOnlyDestinationRemains();
}
} // namespace

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    if (argc != 2) {
        std::cerr << "usage: quantize-output-test DIRECTORY\n";
        return 2;
    }
    output_directory = argv[1];
    return RUN_ALL_TESTS();
}
