#include "vocab.h"
#include <cstdint>
#include <fstream>
#include <limits>
#include <iostream>
#include <stdexcept>
using json = nlohmann::json;
using clap::detail::GemmaVocabulary;

static void expect(const std::vector<int> &actual, const std::vector<int> &wanted) {
    if (actual != wanted) throw std::runtime_error("token mismatch: " + json(actual).dump() + " != " + json(wanted).dump());
}
static json fixture() {
    json m = {{"tokenizer.ggml.model", "gemma4"},
              {"tokenizer.ggml.tokens", {"<pad>", "<eos>", "<bos>", "<unk>", "a", "b", "c", "ab", "bc", "▁", "▁a", "<tag>", "<tag>long", "é", "\n", "\n\n"}},
              {"tokenizer.ggml.token_type", {3,3,3,2,1,1,1,1,1,1,1,4,4,1,1,1}},
              {"tokenizer.ggml.merges", {"b c", "a b", "▁ a", "\n \n"}},
              {"tokenizer.ggml.bos_token_id", 2}, {"tokenizer.ggml.eos_token_id", 1},
              {"tokenizer.ggml.unknown_token_id", 3},
              {"tokenizer.ggml.add_bos_token", true}, {"tokenizer.ggml.add_eos_token", true}};
    const char *hex = "0123456789ABCDEF";
    for (int b = 0; b < 256; ++b) {
        std::string token = "<0x"; token += hex[b >> 4]; token += hex[b & 15]; token += '>';
        m["tokenizer.ggml.tokens"].push_back(token);
        m["tokenizer.ggml.token_type"].push_back(6);
    }
    return m;
}
int main(int argc, char **argv) {
    try {
        auto m = fixture(); GemmaVocabulary v(m, 128);
        expect(v.tokenize("abc",128), {2,4,8,1}); // merges, NOT Unigram/longest matching
        expect(v.tokenize(" a  a ",128), {2,10,9,10,9,1});
        expect(v.tokenize("é",128), {2,13,1});
        expect(v.tokenize("e\xcc\x81",128), {2,16+101,16+204,16+129,1}); // no Unicode normalization
        expect(v.tokenize("🙂",128), {2,16+240,16+159,16+153,16+130,1});
        expect(v.tokenize("\t\r\n\n",128), {2,16+9,16+13,15,1});
        expect(v.tokenize("<bos><tag>longa<eos>",128), {2,2,12,4,1,1});
        expect(v.tokenize("",128), {2,1});
        expect(v.tokenize("aaaa",3), {2,4,1});
        m["tokenizer.ggml.add_bos_token"] = false; m["tokenizer.ggml.add_eos_token"] = false;
        GemmaVocabulary bare(m,128); expect(bare.tokenize("",1), {}); expect(bare.tokenize("ab",1), {7});
        bool rejected = false; try { v.tokenize("a",1); } catch (const std::exception &) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid length accepted");
        for (const char *key : {"tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"}) {
            auto bad = fixture(); bad.erase(key); rejected = false;
            try { GemmaVocabulary invalid(bad,128); } catch (const std::exception &) { rejected = true; }
            if (!rejected) throw std::runtime_error(std::string("missing metadata accepted: ")+key);
        }
        expect(v.tokenize("\0",128), {2,1}); // C string constructor is empty
        expect(v.tokenize(std::string("\0",1),128), {2,16,1});
        auto controls = fixture();
        controls["tokenizer.ggml.add_eos_token"] = false;
        expect(GemmaVocabulary(controls,128).tokenize("a",1), {2});
        controls["tokenizer.ggml.add_bos_token"] = false;
        controls["tokenizer.ggml.add_eos_token"] = true;
        expect(GemmaVocabulary(controls,128).tokenize("a",1), {1});
        auto reject = [&](const json &bad) {
            bool caught = false;
            try { GemmaVocabulary invalid(bad,128); } catch (const std::exception &) { caught = true; }
            if (!caught) throw std::runtime_error("malformed Gemma metadata accepted");
        };
        // Exercise both JSON integer representations before any narrowing to int.
        std::vector<std::string> accepted;
        auto reject_integer = [&](const json &bad, const std::string &label) {
            try { GemmaVocabulary invalid(bad,128); accepted.push_back(label); }
            catch (const std::exception &) {}
        };
        for (const json &value : {json(std::int64_t{4294967297LL}),
                                  json(std::int64_t{-4294967295LL}),
                                  json(std::uint64_t{4294967297ULL}),
                                  json(std::numeric_limits<std::uint64_t>::max())}) {
            auto bad = fixture(); bad["tokenizer.ggml.token_type"][4] = value;
            const auto label = value.type_name() + std::string(":") + value.dump();
            reject_integer(bad, "token_type:" + label);
            for (const char *key : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.unknown_token_id"}) {
                bad = fixture(); bad[key] = value;
                reject_integer(bad, std::string(key) + ":" + label);
                bad["tokenizer.ggml.add_bos_token"] = false;
                bad["tokenizer.ggml.add_eos_token"] = false;
                reject_integer(bad, std::string(key) + ":disabled:" + label);
            }
        }
        auto optional = fixture();
        optional["tokenizer.ggml.add_bos_token"] = false;
        optional["tokenizer.ggml.add_eos_token"] = false;
        for (const char *key : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.unknown_token_id"}) {
            auto bad = optional; bad[key] = -1;
            reject_integer(bad, std::string(key) + ":explicit -1");
            bad[key] = bad["tokenizer.ggml.tokens"].size();
            reject_integer(bad, std::string(key) + ":out of vocabulary");
            optional.erase(key);
        }
        expect(GemmaVocabulary(optional,128).tokenize("a",128), {4});
        if (!accepted.empty()) throw std::runtime_error("integer metadata accepted: " + json(accepted).dump());
        auto bad = fixture(); bad["tokenizer.ggml.bos_token_id"] = 2.5; reject(bad);
        bad = fixture(); bad["tokenizer.ggml.token_type"][4] = 1.5; reject(bad);
        bad = fixture(); bad["tokenizer.ggml.tokens"][4] = ""; reject(bad);
        bad = fixture(); bad["tokenizer.ggml.tokens"][4] = "b"; reject(bad);
        bad = fixture(); bad["tokenizer.ggml.merges"][0] = "missing c"; reject(bad);
        bad = fixture(); bad["tokenizer.ggml.tokens"].erase(16); reject(bad);
        bad = fixture(); bad["tokenizer.ggml.add_space_prefix"] = true; reject(bad);
        bad = fixture(); bad["tokenizer.ggml.model"] = "llama"; reject(bad);
        if (argc == 3) {
            json meta, cases; std::ifstream(argv[1]) >> meta; std::ifstream(argv[2]) >> cases;
            GemmaVocabulary real(meta,8192);
            for (const auto &c : cases) expect(real.tokenize(c.at("text"),8192),c.at("ids").get<std::vector<int>>());
            std::cout << "HF parity cases: " << cases.size() << '\n';
        }
        std::cout << "Gemma tokenizer tests passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
