// Mixed dot adapted from ggml-org/llama.cpp 202c1a229f5a83d1c9d758ed0f2d43760fa4bab0
// ggml/src/ggml-cpu/vec.cpp (MIT, Copyright (c) 2022-2026 The ggml authors).
// Keep this operation-local: changing GGML's F16 traits also changes weight matmuls.
#include "qwen3-attention.h"

// Keep the backend's native half conversions where available. Otherwise use
// the public API, not its private lookup table (unavailable across Windows DLLs).
// Define this before the SIMD headers: their inline vector helpers also widen
// scalar halves and must not capture references to ggml_table_f32_f16.
#if !(defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE) &&                                                        \
      !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)) &&                                 \
    !defined(__POWER9_VECTOR__) && !(defined(__riscv) && defined(__riscv_zfhmin))
#define GGML_CPU_FP16_TO_FP32(x) ggml_fp16_to_fp32(x)
#endif
#include "ggml-cpu/common.h"
#include "ggml-cpu/vec.h"
#include <ggml-cpu.h>

#include <cassert>
#include <cmath>
#include <cstring>

namespace clap::detail {
static void dot_f16_f32(int n, float *GGML_RESTRICT s, size_t bs, const ggml_fp16_t *GGML_RESTRICT x, size_t bx,
                        const float *GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;
    int i = 0;

    // on AVX512 and AVX with F16C, x is widened to F32 in-register and accumulated in F32 over 4 independent sums
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i + 0))),
                               _mm512_loadu_ps(y + i + 0), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i + 16))),
                               _mm512_loadu_ps(y + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i + 32))),
                               _mm512_loadu_ps(y + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i + 48))),
                               _mm512_loadu_ps(y + i + 48), acc3);
    }
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x + i))), _mm512_loadu_ps(y + i),
                               acc0);
    }
    sumf = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
#elif defined(__AVX__) && defined(__F16C__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        acc0 = GGML_F32x8_FMA(acc0, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + 0))),
                              _mm256_loadu_ps(y + i + 0));
        acc1 = GGML_F32x8_FMA(acc1, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + 8))),
                              _mm256_loadu_ps(y + i + 8));
        acc2 = GGML_F32x8_FMA(acc2, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + 16))),
                              _mm256_loadu_ps(y + i + 16));
        acc3 = GGML_F32x8_FMA(acc3, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i + 24))),
                              _mm256_loadu_ps(y + i + 24));
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = GGML_F32x8_FMA(acc0, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i))), _mm256_loadu_ps(y + i));
    }
    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    sumf = _mm_cvtss_f32(r);
#elif defined(GGML_SIMD)
    // convert chunks of x to F32 and reuse the F32 SIMD dot
    // TODO: convert F16 to F32 in-register and fuse the dot (needs a new SIMD macro); also avoid the per-chunk call
    // overhead for tiny k
    float tx[64];
    for (; i + 64 <= n; i += 64) {
        const ggml_fp16_t *GGML_RESTRICT xr = x + i;
        float *GGML_RESTRICT tr = tx;
        for (int j = 0; j < 64; ++j) {
            tr[j] = GGML_CPU_FP16_TO_FP32(xr[j]);
        }
        float v;
        ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot(64, &v, 0, tx, 0, y + i, 0, 1);
        sumf += v;
    }
    if (i < n) {
        const int nr = n - i;
        const ggml_fp16_t *GGML_RESTRICT xr = x + i;
        float *GGML_RESTRICT tr = tx;
        for (int j = 0; j < nr; ++j) {
            tr[j] = GGML_CPU_FP16_TO_FP32(xr[j]);
        }
        float v;
        ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot(nr, &v, 0, tx, 0, y + i, 0, 1);
        sumf += v;
        i = n;
    }
#endif
    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i]) * y[i]);
    }

    *s = sumf;
}

// A small causal, full-sequence specialization, not a second general flash kernel.
// GGML schedules this callback on its CPU pool. Each worker owns complete query rows;
// the output row doubles as the F16 accumulator, so callbacks never allocate/throw.
static void short_attention(ggml_tensor *dst, const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v, int ith,
                            int nth, void *) {
    const int dim = int(q->ne[0]);
    const int64_t tokens = q->ne[1], heads = q->ne[2], group = heads / k->ne[2];
    const float scale = 1.0f / std::sqrt(float(dim));
    const auto row = [](const ggml_tensor *x, int64_t token, int64_t head) {
        return static_cast<const char *>(x->data) + token * x->nb[1] + head * x->nb[2];
    };
    for (int64_t index = ith; index < tokens * heads; index += nth) {
        const int64_t token = index % tokens, head = index / tokens;
        const auto *query = reinterpret_cast<const float *>(row(q, token, head));
        auto *out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + token * dst->nb[1] + head * dst->nb[2]);
        auto *accum = reinterpret_cast<ggml_fp16_t *>(out);
        std::memset(accum, 0, size_t(dim) * sizeof(ggml_fp16_t));
        float maximum = -INFINITY, total = 0.0f;
        for (int64_t key = 0; key <= token; ++key) {
            float score;
            dot_f16_f32(dim, &score, 0, reinterpret_cast<const ggml_fp16_t *>(row(k, key, head / group)), 0, query, 0,
                        1);
            score *= scale;
            float rescale = 1.0f, probability = 1.0f;
            if (score > maximum) {
                rescale = std::exp(maximum - score);
                maximum = score;
                ggml_vec_scale_f16(dim, accum, rescale);
            } else {
                probability = std::exp(score - maximum);
            }
            ggml_vec_mad_f16(dim, accum, reinterpret_cast<const ggml_fp16_t *>(row(v, key, head / group)), probability);
            total = total * rescale + probability;
        }
        const float inverse = 1.0f / total;
        // Widen backwards: writing a float must not overwrite an unread half.
        for (int d = dim - 1; d >= 0; --d) {
            const float value = GGML_CPU_FP16_TO_FP32(accum[d]) * inverse;
            std::memcpy(reinterpret_cast<char *>(out) + size_t(d) * sizeof(float), &value, sizeof(value));
        }
    }
}

ggml_tensor *qwen3_attention_cpu(ggml_context *ctx, ggml_tensor *q, ggml_tensor *k, ggml_tensor *v, ggml_tensor *mask) {
    // Match the pinned CPU backend's tiled dispatch. That path already preserves
    // Q as F32; only its short-query path needs the mixed-dot specialization.
    bool tiled = q->ne[1] >= int64_t(ggml_fa_tile_config::Q);
#if defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#if defined(__ARM_FEATURE_SVE)
    const int64_t lanes = svcntw();
#else
    const int64_t lanes = GGML_F32_EPR;
#endif
    tiled &= v->ne[0] % lanes == 0;
#endif
    if (tiled) {
        auto *out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(q->ne[0])), 0, 0);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        return out;
    }
    auto *out = ggml_map_custom3(ctx, q, k, v, short_attention, GGML_N_TASKS_MAX, nullptr);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}
} // namespace clap::detail
