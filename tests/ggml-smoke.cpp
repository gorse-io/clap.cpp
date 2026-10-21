#include <ggml.h>
#include <gtest/gtest.h>
#include <memory>

TEST(GgmlSmoke, AllocatesAndAccessesTensor) {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    ASSERT_NE(nullptr, ctx.get());

    ggml_tensor *tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
    ASSERT_NE(nullptr, tensor);
    float *data = ggml_get_data_f32(tensor);
    ASSERT_NE(nullptr, data);
    data[3] = 2.0f;
    EXPECT_EQ(4, ggml_nelements(tensor));
    EXPECT_FLOAT_EQ(2.0f, data[3]);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
