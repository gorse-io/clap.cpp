#include "model.h"
#include "models/model-policy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace clap::detail {
namespace {
struct TensorData {
    const char *name;
    std::vector<int64_t> dimensions;
    ggml_type type = GGML_TYPE_F32;
    float value = 0;
};

class TensorContractTest : public testing::Test {
  protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory = std::filesystem::temp_directory_path() / ("clap-tensor-contract-" + std::to_string(stamp));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
    }
    void TearDown() override {
        loader.reset();
        std::filesystem::remove_all(directory);
    }
    void load(std::initializer_list<TensorData> tensors, const char *architecture = "clap2", bool inference = true,
              const char *metadata_key = nullptr) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({1024 * 1024, nullptr, false}),
                                                                    ggml_free);
        ASSERT_NE(context, nullptr);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
        ASSERT_NE(file, nullptr);
        if (architecture)
            gguf_set_val_str(file.get(), "general.architecture", architecture);
        if (metadata_key)
            gguf_set_val_str(file.get(), metadata_key, "fixture");
        for (const auto &spec : tensors) {
            auto *tensor = ggml_new_tensor(context.get(), spec.type, static_cast<int>(spec.dimensions.size()),
                                           spec.dimensions.data());
            ggml_set_name(tensor, spec.name);
            std::memset(tensor->data, 0, ggml_nbytes(tensor));
            if (spec.value != 0) {
                if (spec.type == GGML_TYPE_F32)
                    std::fill_n(static_cast<float *>(tensor->data), ggml_nelements(tensor), spec.value);
                else {
                    ASSERT_EQ(spec.type, GGML_TYPE_F16);
                    std::fill_n(static_cast<ggml_fp16_t *>(tensor->data), ggml_nelements(tensor),
                                ggml_fp32_to_fp16(spec.value));
                }
            }
            gguf_add_tensor(file.get(), tensor);
        }
        const auto path = (directory / "model.gguf").string();
        ASSERT_TRUE(gguf_write_to_file(file.get(), path.c_str(), false));
        loader = std::make_unique<ModelLoader>();
        load_model_weights(*loader, path, inference);
    }
    void truncate_tensor(int64_t index) {
        loader.reset();
        const auto path = directory / "model.gguf";
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(
            gguf_init_from_file(path.string().c_str(), {true, nullptr}), gguf_free);
        ASSERT_NE(file, nullptr);
        std::filesystem::resize_file(path,
                                     gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), index) + 1);
    }
    std::filesystem::path directory;
    std::unique_ptr<ModelLoader> loader;
};

TEST_F(TensorContractTest, RequiredTensorReturnsTheValidatedMappedTensor) {
    load({{"dense.weight", {3, 2}}});
    auto *tensor = loader->create_tensor("dense.weight", {3, 2});
    EXPECT_EQ(tensor, loader->weight("dense.weight"));
    EXPECT_NE(tensor->data, nullptr);
    EXPECT_EQ(tensor->type, GGML_TYPE_F32);
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, MissingRequiredTensorRetainsItsDiagnostic) {
    load({{"present", {3}}});
    try {
        loader->create_tensor("missing", {3});
        FAIL() << "missing required tensor was accepted";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ(error.what(), "CLAP: missing tensor missing");
    }
}

TEST_F(TensorContractTest, OptionalMeansAbsentIsAllowedNotValidationIsSkipped) {
    load({{"present", {3, 2}}});
    EXPECT_EQ(loader->create_tensor("absent", {3}, TensorRequirement::Optional), nullptr);
    EXPECT_THROW(loader->create_tensor("present", {2, 3}, TensorRequirement::Optional), std::runtime_error);
    EXPECT_EQ(loader->create_tensor("present", {3, 2}, TensorRequirement::Optional), loader->weight("present"));
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, RejectsUnexpectedNonSingletonAxes) {
    load({{"ranked", {3, 2, 2}}});
    EXPECT_THROW(loader->create_tensor("ranked", {3, 2}), std::runtime_error);
    EXPECT_NO_THROW(loader->create_tensor("ranked", {3, 2, 2}));
}

TEST_F(TensorContractTest, AcceptsOmittedSingletonAxes) {
    load({{"vector", {3, 1, 1}}});
    EXPECT_EQ(loader->create_tensor("vector", {3}), loader->weight("vector"));
}

TEST_F(TensorContractTest, RejectsInvalidContractDimensions) {
    load({{"vector", {3}}});
    EXPECT_THROW(loader->create_tensor("vector", {}), std::runtime_error);
    EXPECT_THROW(loader->create_tensor("vector", {0}), std::runtime_error);
    EXPECT_THROW(loader->create_tensor("vector", {-3}), std::runtime_error);
    EXPECT_THROW(loader->create_tensor("vector", {3, 1, 1, 1, 1}), std::runtime_error);
}

TEST_F(TensorContractTest, FailedValidationDoesNotCompleteCoverage) {
    load({{"vector", {3}}});
    EXPECT_THROW(loader->create_tensor("vector", {2}), std::runtime_error);
    EXPECT_THROW(loader->validate_tensor_coverage(), std::runtime_error);
    EXPECT_NO_THROW(loader->create_tensor("vector", {3}));
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, UnrequestedTensorStillFailsStrictCoverage) {
    load({{"used", {3}}, {"unused", {3}}});
    loader->create_tensor("used", {3});
    EXPECT_THROW(loader->validate_tensor_coverage(), std::runtime_error);
}

TEST_F(TensorContractTest, NamespaceViewsValidateTheSourceAndPreserveIdentity) {
    load({{"mert.projection.weight", {3, 2}}});
    ModelLoader view(*loader, "mert.", "clamp3.");
    EXPECT_NO_THROW(view.validate_tensor_coverage());
    EXPECT_THROW(view.create_tensor("projection.weight", {2, 3}), std::runtime_error);
    EXPECT_THROW(loader->validate_tensor_coverage(), std::runtime_error);
    EXPECT_EQ(view.create_tensor("projection.weight", {3, 2}), loader->weight("mert.projection.weight"));
    EXPECT_EQ(view.create_tensor("absent", {3}, TensorRequirement::Optional), nullptr);
    EXPECT_THROW(view.create_tensor("projection.weight", {2, 3}), std::runtime_error);
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, LinearContractIncludesOnlyDeclaredBias) {
    load({{"dense.weight", {3, 2}}, {"dense.bias", {2}}, {"biasless.weight", {3, 2}}});
    const auto dense = create_linear(*loader, "dense", 3, 2);
    EXPECT_EQ(dense.weight, loader->weight("dense.weight"));
    EXPECT_EQ(dense.bias, loader->weight("dense.bias"));
    const auto biasless = create_linear(*loader, "biasless", 3, 2, false);
    EXPECT_EQ(biasless.bias, nullptr);
    EXPECT_THROW(create_linear(*loader, "biasless", 3, 2), std::runtime_error);
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, NormalizationContractValidatesBothAffineVectors) {
    load({{"norm.weight", {3}}, {"norm.bias", {2}}});
    EXPECT_THROW(create_norm(*loader, "norm", 3), std::runtime_error);
}

TEST_F(TensorContractTest, LayerContractPreservesConvolutionAndGrnBiasGeometry) {
    load({{"conv.weight", {3, 1, 2}}, {"conv.bias", {2}}, {"grn.weight", {8, 1, 1}}, {"grn.bias", {8, 1, 1}}});
    const auto convolution = create_layer(*loader, "conv", {3, 1, 2}, {2});
    const auto grn = create_layer(*loader, "grn", {8, 1, 1}, {8, 1, 1});
    EXPECT_EQ(convolution.bias, loader->weight("conv.bias"));
    EXPECT_EQ(grn.bias, loader->weight("grn.bias"));
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, BatchNormalizationIncludesRunningStatistics) {
    load({{"bn.weight", {3}}, {"bn.bias", {3}}, {"bn.running_mean", {3}}, {"bn.running_var", {3}}});
    const auto batch = BatchNormWeights::create(*loader, "bn", 3);
    EXPECT_EQ(batch.mean, loader->weight("bn.running_mean"));
    EXPECT_EQ(batch.variance, loader->weight("bn.running_var"));
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, FeatureEmbeddingsDoNotRequireAWordTable) {
    load({{"emb.position_embeddings.weight", {3, 4}},
          {"emb.token_type_embeddings.weight", {3, 1}},
          {"emb.LayerNorm.weight", {3}},
          {"emb.LayerNorm.bias", {3}}});
    const auto embeddings = create_embeddings(*loader, "emb", 3, 0, 4, 1, false);
    EXPECT_EQ(embeddings.word, nullptr);
    EXPECT_EQ(embeddings.position, loader->weight("emb.position_embeddings.weight"));
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, RawLoaderPreservesStorageInsteadOfSelectingMuqPrecision) {
    load({{"dense.weight", {3, 2}, GGML_TYPE_F16}}, "muq");
    ModelLoader raw;
    raw.load((directory / "model.gguf").string());
    EXPECT_EQ(raw.weight("dense.weight")->type, GGML_TYPE_F16);
    EXPECT_FALSE(raw.f32_activation_compute());
}

TEST_F(TensorContractTest, RawLoaderDoesNotInferStrictCoverageFromArchitectureMetadata) {
    load({{"unused", {3}}});
    ModelLoader raw;
    raw.load((directory / "model.gguf").string());
    EXPECT_NO_THROW(raw.validate_tensor_coverage());
}

TEST_F(TensorContractTest, RawLoaderDoesNotImposeMulanVectorContracts) {
    load({{"norm.weight", {3}, GGML_TYPE_F16}}, "muq_mulan");
    ModelLoader raw;
    raw.load((directory / "model.gguf").string());
    EXPECT_NO_THROW(raw.create_tensor("norm.weight", {3}));
}

TEST_F(TensorContractTest, MulanVectorStorageRestrictionIsPreserved) {
    load({{"norm.weight", {3}, GGML_TYPE_F16}}, "muq_mulan");
    try {
        loader->create_tensor("norm.weight", {2});
        FAIL() << "F16 vector contract was accepted";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ(error.what(), "CLAP: MuLan vectors require F32: norm.weight");
    }
    EXPECT_THROW(loader->create_tensor("norm.weight", {3}, TensorRequirement::Optional), std::runtime_error);
    EXPECT_THROW(loader->validate_tensor_coverage(), std::runtime_error);
}

TEST_F(TensorContractTest, ModelStorageModePreservesMuqBytesAndInferencePromotesThem) {
    load({{"dense.weight", {3, 2}, GGML_TYPE_F16, 0.25f}}, "muq", false);
    const auto *stored = loader->weight("dense.weight");
    ASSERT_EQ(stored->type, GGML_TYPE_F16);
    EXPECT_EQ(stored->nb[0], sizeof(ggml_fp16_t));
    ModelLoader raw;
    raw.load((directory / "model.gguf").string());
    EXPECT_EQ(std::memcmp(stored->data, raw.weight("dense.weight")->data, ggml_nbytes(stored)), 0);
    ModelLoader inference;
    load_model_weights(inference, (directory / "model.gguf").string());
    const auto *promoted = inference.weight("dense.weight");
    EXPECT_EQ(promoted->type, GGML_TYPE_F32);
    EXPECT_EQ(promoted->nb[0], sizeof(float));
    EXPECT_EQ(static_cast<const float *>(promoted->data)[0], 0.25f);
    EXPECT_TRUE(inference.f32_activation_compute());
}

TEST_F(TensorContractTest, MulanPromotesOnlyItsBackboneAndNeverInStorageMode) {
    load({{"conv.kernel", {3, 2}, GGML_TYPE_F16}, {"text.dense.weight", {3, 2}, GGML_TYPE_F16}}, "muq_mulan");
    EXPECT_EQ(loader->weight("conv.kernel")->type, GGML_TYPE_F32);
    EXPECT_EQ(loader->weight("text.dense.weight")->type, GGML_TYPE_F16);
    ModelLoader storage;
    load_model_weights(storage, (directory / "model.gguf").string(), false);
    EXPECT_EQ(storage.weight("conv.kernel")->type, GGML_TYPE_F16);
    EXPECT_EQ(storage.weight("text.dense.weight")->type, GGML_TYPE_F16);
}

TEST_F(TensorContractTest, BundledCoverageIsIndependentOfActivationPrecisionAndRootOnly) {
    load({{"mert.projection.weight", {3, 2}}}, "clamp3", true, "clamp3.audio_frontend.descendant");
    EXPECT_FALSE(loader->f32_activation_compute());
    EXPECT_THROW(loader->validate_tensor_coverage(), std::runtime_error);
    ModelLoader view(*loader, "mert.", "clamp3.");
    EXPECT_NO_THROW(view.validate_tensor_coverage());
    view.create_tensor("projection.weight", {3, 2});
    EXPECT_NO_THROW(loader->validate_tensor_coverage());
}

TEST_F(TensorContractTest, NeutralModelPoliciesDoNotRequireConfigurationOrStrictCoverage) {
    for (const char *architecture : {"clap", "clamp3", "unknown", static_cast<const char *>(nullptr)}) {
        load({{"unused", {3}}}, architecture);
        EXPECT_FALSE(loader->f32_activation_compute());
        EXPECT_NO_THROW(loader->validate_tensor_coverage());
    }
}

TEST_F(TensorContractTest, MulanStorageModeStillRejectsInvalidContentOnExtraTensors) {
    EXPECT_THROW(load({{"extra", {3}, GGML_TYPE_F32, std::numeric_limits<float>::quiet_NaN()}}, "muq_mulan", false),
                 std::runtime_error);
    EXPECT_THROW(load({{"bn.running_var", {3}, GGML_TYPE_F32, -1}}, "muq_mulan", false), std::runtime_error);
    EXPECT_THROW(load({{"frontend.window", {3}, GGML_TYPE_F16}}, "muq_mulan", false), std::runtime_error);
    EXPECT_THROW(load({{"conformer.norm.weight", {3}, GGML_TYPE_F16}}, "muq_mulan", false), std::runtime_error);
}

TEST_F(TensorContractTest, ArchitectureTypeErrorPrecedesSameTensorTruncation) {
    EXPECT_THROW(load({{"dense.weight", {32, 2}, GGML_TYPE_Q8_0}}), std::runtime_error);
    truncate_tensor(0);
    const auto path = directory / "model.gguf";
    ModelLoader raw;
    EXPECT_THROW(raw.load(path.string()), std::runtime_error);
    ModelLoader truncated;
    try {
        load_model_weights(truncated, path.string());
        FAIL() << "quantized CLAP2 tensor was accepted";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ(error.what(), "CLAP: CLAP2 block-quantized weights are not supported; use F32/F16/BF16");
    }
}

TEST_F(TensorContractTest, EarlierContentErrorPrecedesLaterTensorTruncation) {
    EXPECT_THROW(
        load({{"extra", {3}, GGML_TYPE_F32, std::numeric_limits<float>::quiet_NaN()}, {"later", {3}}}, "muq_mulan"),
        std::runtime_error);
    truncate_tensor(1);
    const auto path = directory / "model.gguf";
    ModelLoader raw;
    EXPECT_THROW(raw.load(path.string()), std::runtime_error);
    ModelLoader truncated;
    try {
        load_model_weights(truncated, path.string());
        FAIL() << "non-finite MuLan extra tensor was accepted";
    } catch (const std::runtime_error &error) {
        EXPECT_STREQ(error.what(), "CLAP: non-finite MuLan tensor: extra");
    }
}
} // namespace
} // namespace clap::detail
