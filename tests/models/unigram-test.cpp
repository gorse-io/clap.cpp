#include "vocab.h"
#include "../oracle-gtest.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static std::string tokenizer_path, cases_path;

class UnigramTokenizer : public testing::Test {
protected:
    static nlohmann::json tokenizer, cases;
    static void SetUpTestSuite() {
        std::ifstream input(tokenizer_path);
        ASSERT_TRUE(input.is_open()) << "cannot open tokenizer: " << tokenizer_path;
        input >> tokenizer;
        std::ifstream reference(cases_path);
        ASSERT_TRUE(reference.is_open()) << "cannot open cases: " << cases_path;
        reference >> cases;
    }
};
nlohmann::json UnigramTokenizer::tokenizer, UnigramTokenizer::cases;

TEST_F(UnigramTokenizer, OracleParity) {
    clap::detail::UnigramVocabulary vocab(tokenizer, 0, 2, 3, 4096);
    for (const auto &test : cases) {
        SCOPED_TRACE(test.at("text").get<std::string>());
        const auto actual = vocab.tokenize(test.at("text").get<std::string>(), test.value("max_length", 4096));
        const auto expected = test.at("ids").get<std::vector<int>>();
        ASSERT_EQ(actual, expected);
    }
}

TEST_F(UnigramTokenizer, RejectsUnsupportedMetadata) {
    auto reject = [](const nlohmann::json &invalid) {
        EXPECT_THROW((clap::detail::UnigramVocabulary(invalid, 0, 2, 3, 4096)), std::exception)
            << "unsupported tokenizer metadata was accepted";
    };
    auto invalid = tokenizer;
    invalid["pre_tokenizer"] = nullptr; reject(invalid);
    invalid = tokenizer;
    invalid["pre_tokenizer"]["pretokenizers"][1]["prepend_scheme"] = "never"; reject(invalid);
    invalid = tokenizer;
    invalid["post_processor"] = nullptr; reject(invalid);
    invalid = tokenizer;
    invalid["post_processor"]["single"] = nlohmann::json::array(); reject(invalid);
    invalid = tokenizer;
    invalid["model"]["unk_id"] = 4; reject(invalid);
    invalid = tokenizer;
    invalid["model"]["byte_fallback"] = true; reject(invalid);
    if (!HasFailure())
        std::cout << "Unigram tokenizer parity passed: " << cases.size() << " cases; invalid metadata rejected\n";
}

int main(int argc, char **argv) {
    oracle_test::initialize(&argc, argv);
    if (argc != 3 && !GTEST_FLAG_GET(list_tests)) {
        std::cerr << "usage: clap-unigram-test TOKENIZER.json CASES.json\n";
        return 1;
    }
    if (argc == 3) { tokenizer_path = argv[1]; cases_path = argv[2]; }
    return RUN_ALL_TESTS();
}
