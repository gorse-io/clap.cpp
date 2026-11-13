/*
 * Copyright 2026 clap.cpp Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "context.h"
#include "model.h"

#include <gguf.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace clap::detail;
using json = nlohmann::json;

// Write real GGUF metadata, tensor descriptors and payloads through the GGUF API.
// Fixtures live on the build/test temporary filesystem, never in the source tree.
struct Fixture {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file{gguf_init_empty(), gguf_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors{
        ggml_init({1024 * ggml_tensor_overhead(), nullptr, true}), ggml_free};
    std::vector<std::unique_ptr<float[]>> data;
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("clap-contract-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    ~Fixture() { std::filesystem::remove(path); }
    void metadata(const std::string &key, const std::string &value) {
        gguf_set_val_str(file.get(), key.c_str(), value.c_str());
    }
    void tensor(const std::string &name, std::initializer_list<int64_t> dimensions) {
        std::vector<int64_t> dims(dimensions);
        auto *t = ggml_new_tensor(tensors.get(), GGML_TYPE_F32, int(dims.size()), dims.data());
        ggml_set_name(t, name.c_str());
        auto values = std::make_unique<float[]>(ggml_nelements(t));
        // Nonconstant deterministic weights also permit actual encoder characterization.
        for (int64_t i = 0; i < ggml_nelements(t); ++i)
            values[i] = float((i % 7) - 3) * 0.03f;
        t->data = values.get();
        data.push_back(std::move(values));
        gguf_add_tensor(file.get(), t);
    }
    void norm(const std::string &name, int width) {
        tensor(name + ".weight", {width});
        tensor(name + ".bias", {width});
    }
    void linear(const std::string &name, int input, int output, bool bias = true) {
        tensor(name + ".weight", {input, output});
        if (bias)
            tensor(name + ".bias", {output});
    }
    void bert(const std::string &p, int width) {
        for (const auto *name : {"query", "key", "value"})
            linear(p + ".attention.self." + name, width, width);
        linear(p + ".attention.output.dense", width, width);
        norm(p + ".attention.output.LayerNorm", width);
        norm(p + ".output.LayerNorm", width);
        linear(p + ".intermediate.dense", width, width * 2);
        linear(p + ".output.dense", width * 2, width);
    }
    void mert(int stable) {
        metadata("general.architecture", "mert");
        json config = {{"model_type", "mert_model"},
                       {"sample_rate", 24000},
                       {"conv_bias", true},
                       {"feat_extract_norm", "group"},
                       {"hidden_act", "gelu"},
                       {"feat_extract_activation", "gelu"},
                       {"hidden_size", 4},
                       {"intermediate_size", 8},
                       {"num_hidden_layers", 1},
                       {"num_attention_heads", 1},
                       {"num_conv_pos_embeddings", 4},
                       {"num_conv_pos_embedding_groups", 2},
                       {"conv_dim", {4}},
                       {"conv_kernel", {3}},
                       {"conv_stride", {2}},
                       {"layer_norm_eps", 1e-5}};
        if (stable >= 0)
            config["do_stable_layer_norm"] = stable != 0;
        metadata("mert.config", config.dump());
        tensor("frontend.0.conv.weight", {3, 1, 4});
        tensor("frontend.0.conv.bias", {4});
        norm("frontend.0.layer_norm", 4);
        norm("projection.layer_norm", 4);
        linear("projection.projection", 4, 4);
        tensor("position.weight", {4, 2, 4});
        tensor("position.bias", {4});
        norm("encoder.layer_norm", 4);
        for (const auto *name : {"q_proj", "k_proj", "v_proj", "out_proj"})
            linear(std::string("layers.0.attention.") + name, 4, 4);
        norm("layers.0.layer_norm", 4);
        norm("layers.0.final_layer_norm", 4);
        linear("layers.0.ffn.in", 4, 8);
        linear("layers.0.ffn.out", 8, 4);
    }
    void write() { ASSERT_TRUE(gguf_write_to_file(file.get(), path.string().c_str(), false)); }
    void clap(bool scales = false, bool malformed_scale = false) {
        metadata("general.architecture", "clap");
        json audio = {{"enable_fusion", false},
                      {"depths", {1, 1, 1, 1}},
                      {"num_attention_heads", {1, 1, 1, 1}},
                      {"patch_size", 4},
                      {"patch_stride", {4, 4}},
                      {"patch_embed_input_channels", 1},
                      {"qkv_bias", true},
                      {"enable_patch_layer_norm", true},
                      {"flatten_patch_embeds", true},
                      {"hidden_act", "gelu"},
                      {"projection_hidden_act", "relu"},
                      {"spec_size", 32},
                      {"num_mel_bins", 32},
                      {"window_size", 1},
                      {"patch_embeds_hidden_size", 2},
                      {"hidden_size", 16},
                      {"mlp_ratio", 2},
                      {"projection_dim", 4}};
        json text = {{"hidden_act", "gelu"},   {"projection_hidden_act", "relu"},
                     {"hidden_size", 4},       {"num_attention_heads", 1},
                     {"num_hidden_layers", 1}, {"type_vocab_size", 1},
                     {"vocab_size", 4},        {"max_position_embeddings", 8},
                     {"intermediate_size", 8}, {"projection_dim", 4},
                     {"pad_token_id", 1},      {"bos_token_id", 0},
                     {"eos_token_id", 2},      {"layer_norm_eps", 1e-5}};
        metadata("clap.config",
                 json({{"model_type", "clap"}, {"projection_dim", 4}, {"audio_config", audio}, {"text_config", text}})
                     .dump());
        metadata("clap.tokenizer", R"({"model":{"type":"BPE","vocab":{"a":3},"merges":[]}})");
        tensor("audio.patch_embed.proj.weight", {4, 4, 1, 2});
        tensor("audio.patch_embed.proj.bias", {2});
        norm("audio.patch_embed.norm", 2);
        norm("audio.batch_norm", 32);
        tensor("audio.batch_norm.running_mean", {32});
        tensor("audio.batch_norm.running_var", {32});
        int dim = 2;
        for (int stage = 0; stage < 4; ++stage) {
            const auto p = "audio.layers." + std::to_string(stage);
            const auto b = p + ".blocks.0";
            norm(b + ".layernorm_before", dim);
            norm(b + ".layernorm_after", dim);
            for (const auto *name : {"query", "key", "value"})
                linear(b + ".attention.self." + name, dim, dim);
            tensor(b + ".attention.self.rel_bias", {1, 1});
            linear(b + ".attention.output.dense", dim, dim);
            linear(b + ".intermediate.dense", dim, dim * 2);
            linear(b + ".output.dense", dim * 2, dim);
            if (stage < 3) {
                norm(p + ".downsample.norm", dim * 4);
                linear(p + ".downsample.reduction", dim * 4, dim * 2, false);
            }
            dim *= 2;
        }
        norm("audio.norm", 16);
        tensor("text.embeddings.word_embeddings.weight", {4, 4});
        tensor("text.embeddings.position_embeddings.weight", {4, 8});
        tensor("text.embeddings.token_type_embeddings.weight", {4, 1});
        norm("text.embeddings.LayerNorm", 4);
        bert("text.encoder.layer.0", 4);
        linear("text.pooler.dense", 4, 4);
        linear("text_proj.linear1", 4, 4);
        linear("text_proj.linear2", 4, 4);
        linear("audio_proj.linear1", 16, 4);
        linear("audio_proj.linear2", 4, 4);
        if (scales) {
            tensor("logit_scale_a", {malformed_scale ? 2 : 1});
            tensor("logit_scale_t", {1});
        }
    }
};
clap::RuntimeOptions cpu_options() {
    clap::RuntimeOptions options;
    options.device = "cpu";
    options.threads = 1;
    return options;
}

TEST(ModelContract, MertStableLayerNormDefaultMatchesFalse) {
    std::vector<std::vector<float>> outputs;
    for (int stable : {-1, 0, 1}) {
        Fixture fixture;
        fixture.mert(stable);
        fixture.write();
        auto model = std::make_shared<Model>(fixture.path.string(), cpu_options());
        Context context(model);
        std::vector<float> pcm(31);
        for (size_t i = 0; i < pcm.size(); ++i)
            pcm[i] = float((i * i) % 13) * 0.1f;
        const auto frames = model->architecture->audio_frames(context, pcm);
        EXPECT_EQ(frames.frames, 15);
        EXPECT_EQ(frames.dimension, 4);
        RecordProperty("mert_stable_" + std::to_string(stable + 1), json(frames.values).dump());
        EXPECT_EQ(frames.values, model->architecture->audio_frames(context, pcm).values);
        outputs.push_back(frames.values);
    }
    EXPECT_EQ(outputs[0], outputs[1]);
    EXPECT_NE(outputs[1], outputs[2]);
}
TEST(ModelContract, ClapAcceptsAbsentOrScalarTrainingScales) {
    for (bool scales : {false, true}) {
        Fixture fixture;
        fixture.clap(scales);
        fixture.write();
        EXPECT_NO_THROW(Model(fixture.path.string(), cpu_options()));
    }
}
TEST(ModelContract, ClapRejectsUnknownTensorThroughModelConstruction) {
    Fixture fixture;
    fixture.clap(true);
    fixture.tensor("unexpected.weight", {2});
    fixture.write();
    try {
        Model model(fixture.path.string(), cpu_options());
        FAIL() << "CLAP accepted an undeclared checkpoint tensor";
    } catch (const std::runtime_error &e) {
        EXPECT_EQ(std::string(e.what()), "CLAP: unrecognized model tensor: unexpected.weight");
    }
}
TEST(ModelContract, ClapRejectsMalformedOptionalTrainingScale) {
    Fixture fixture;
    fixture.clap(true, true);
    fixture.write();
    try {
        Model model(fixture.path.string(), cpu_options());
        FAIL() << "CLAP accepted a nonscalar training scale";
    } catch (const std::runtime_error &e) {
        EXPECT_EQ(std::string(e.what()), "CLAP: wrong tensor shape: logit_scale_a");
    }
}
TEST(ModelContract, AllArchitecturePoliciesRequireTensorCoverage) {
    for (const auto *architecture : {"clap", "clap2", "clamp3", "mert", "mert2", "muq", "muq_mulan"}) {
        SCOPED_TRACE(architecture);
        Fixture fixture;
        fixture.metadata("general.architecture", architecture);
        fixture.tensor("unexpected.weight", {2});
        fixture.write();
        ModelLoader loader(cpu_options());
        load_model_weights(loader, fixture.path.string());
        const std::string name = architecture;
        EXPECT_EQ(loader.f32_activation_compute(), name != "clap" && name != "clamp3");
        try {
            loader.validate_tensor_coverage();
            ADD_FAILURE() << "architecture policy accepted an undeclared tensor";
        } catch (const std::runtime_error &e) {
            EXPECT_EQ(std::string(e.what()), (name == "muq_mulan" ? "CLAP: unrecognized MuLan tensor: "
                                                                  : "CLAP: unrecognized model tensor: ") +
                                                 std::string("unexpected.weight"));
        }
    }
}
TEST(ModelContract, StorageLoaderRemainsNeutralForKnownAndUnknownArchitectures) {
    for (const auto *architecture : {"clap", "clamp3", "unknown"}) {
        Fixture fixture;
        fixture.metadata("general.architecture", architecture);
        fixture.tensor("storage.only", {2});
        fixture.write();
        for (bool inference : {false, true}) {
            ModelLoader loader;
            loader.load(fixture.path.string(), inference);
            EXPECT_NO_THROW(loader.validate_tensor_coverage());
            EXPECT_FALSE(loader.f32_activation_compute());
            EXPECT_EQ(loader.weight("storage.only")->type, GGML_TYPE_F32);
        }
        if (std::string(architecture) == "unknown") {
            ModelLoader loader;
            load_model_weights(loader, fixture.path.string());
            EXPECT_NO_THROW(loader.validate_tensor_coverage());
        }
    }
}
} // namespace
