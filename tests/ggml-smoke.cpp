#include <ggml.h>

#include <cstdio>

int main() {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fputs("Failed to initialize ggml context\n", stderr);
        return 1;
    }

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    float * data = ggml_get_data_f32(tensor);
    data[3] = 2.0f;
    const bool valid = ggml_nelements(tensor) == 4 && data[3] == 2.0f;
    ggml_free(ctx);
    return valid ? 0 : 1;
}
