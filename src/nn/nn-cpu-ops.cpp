#ifdef _WIN32
    #define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#endif
#include "nn-cpu-ops.hpp"
#include "nn-quants.hpp"
#include "llamafile/sgemm.hpp"

#define DEBUG_OP_INPUT_OUTPUT false

#if DEBUG_OP_INPUT_OUTPUT
    #define DEBUG_VECTOR(context, suffix, vec) \
        if (threadIndex == 0) \
            printf("%20s.%6s: %f %f %f %f\n", context->name, suffix, vec[0], vec[1], vec[2], vec[3]);
    
    #define DEBUG_SCALAR(context, suffix, scalar) \
        if (threadIndex == 0) \
            printf("%20s.%6s: %f\n", context->name, suffix, scalar);
#else
    #define DEBUG_VECTOR(context, suffix, vec)
    #define DEBUG_SCALAR(context, suffix, scalar)
#endif

#if defined(__AVX2__)
static inline float hsum_F8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif

#if defined(__ARM_NEON)
static inline float32x4_t exp_F32_neon(float32x4_t x) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718056f);
    const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504089f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.1666666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166666667f);
    const float32x4_t c5 = vdupq_n_f32(0.008333333333f);

    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    float32x4_t kf = vaddq_f32(vmulq_f32(x, inv_ln2), vdupq_n_f32(0.5f));
    int32x4_t k = vcvtq_s32_f32(kf);
    kf = vcvtq_f32_s32(k);

    float32x4_t f = vmlsq_f32(x, kf, ln2);
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f3, f);
    float32x4_t f5 = vmulq_f32(f4, f);
    float32x4_t p = c1;
    p = vaddq_f32(p, f);
    p = vaddq_f32(p, vmulq_f32(c2, f2));
    p = vaddq_f32(p, vmulq_f32(c3, f3));
    p = vaddq_f32(p, vmulq_f32(c4, f4));
    p = vaddq_f32(p, vmulq_f32(c5, f5));

    int32x4_t pow2k = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    float32x4_t two_k = vreinterpretq_f32_s32(pow2k);
    return vmulq_f32(p, two_k);
}
#endif

static float rms_F32(const float *x, const unsigned int size, const float epsilon) {
    float ss;
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fsq;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int j = 0; j < size; j += 4) {
        fsq = vld1q_f32(&x[j]);
        fs = vmlaq_f32(fs, fsq, fsq);
    }
    ss = vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int j = 0; j < size; j += 8) {
        a = _mm256_loadu_ps(&x[j]);
        u = _mm256_fmadd_ps(a, a, u);
    }
    ss = hsum_F8(u);
#else
    ss = 0;
    for (unsigned int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
#endif
    ss /= size;
    ss += epsilon;
    ss = 1.0f / sqrtf(ss);
    return ss;
}

static void rmsNorm_F32(float *output, const float *x, const float rms, const float *w, const NnSize size, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    assert((start - end) % 4 == 0);
    float32x4_t fw;
    float32x4_t fx;
    float32x4_t fss = vmovq_n_f32(rms);
    for (unsigned int i = start; i < end; i += 4) {
        fw = vld1q_f32(&w[i]);
        fx = vld1q_f32(&x[i]);
        fx = vmulq_f32(fx, fw);
        fx = vmulq_f32(fx, fss);
        vst1q_f32(&output[i], fx);
    }
#else
    for (unsigned int i = start; i < end; i++) {
        output[i] = w[i] * (rms * x[i]);
    }
#endif
}

static void rmsNorm_Q80_F32_F32(float *output, const NnBlockQ80 *x, const float rms, const float *w, const NnSize size, const NnSize nThreads, const NnSize threadIndex) {
    assert(size % Q80_BLOCK_SIZE == 0);
    const NnSize nBlocks = size / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

    for (NnSize i = start; i < end; i++) {
        float d = convertF16toF32(x[i].d);
        for (NnSize j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnSize k = i * Q80_BLOCK_SIZE + j;
            output[k] = w[k] * (rms * d * x[i].qs[j]);
        }
    }
}

static void matmul_F32_F32_F32(float *output, const float *x, const float *w, const NnSize n, const NnSize d, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    unsigned int i, j;
#if defined(__ARM_NEON)
    assert(n % 4 == 0);
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (i = start; i < end; i++) {
        z = vmovq_n_f32(0);
        for (j = 0; j < n; j += 4) {
            q = vld1q_f32(&x[j]);
            p = vld1q_f32(&w[i * n + j]);
            z = vfmaq_f32(z, q, p);
        }
        output[i] = vaddvq_f32(z);
    }
#elif defined(__AVX2__)
    assert(n % 8 == 0);
    __m256 a0, b0, u;
    for (i = start; i < end; i++) {
        u = _mm256_set1_ps(0.0f);
        for (j = 0; j < n; j += 8) {
            a0 = _mm256_loadu_ps(&x[j]);
            b0 = _mm256_loadu_ps(&w[i * n + j]);
            u = _mm256_fmadd_ps(a0, b0, u);
        }
        output[i] = hsum_F8(u);
    }
#else
    for (i = start; i < end; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        output[i] = val;
    }
#endif
}

static void matmul_F32_Q40_F32(float *output, const float *x, const NnBlockQ40 *w, const NnSize n, const NnSize d, const NnSize nThreads, const NnSize threadIndex) {
    assert(n % Q40_BLOCK_SIZE == 0);
    const NnSize nBlocks = n / Q40_BLOCK_SIZE;
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);

#if defined(__ARM_NEON)
    for (NnSize i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnSize j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *b = &w[i * nBlocks + j];
            const float d_scale = convertF16toF32(b->d);
            const float *x_block = x + j * Q40_BLOCK_SIZE;

            float32x4_t sumv = vdupq_n_f32(0.0f);

            for (NnSize k = 0; k < Q40_BLOCK_SIZE / 2; k += 8) {
                uint8x8_t qs8 = vld1_u8(b->qs + k);

                uint8x8_t w0_u8 = vand_u8(qs8, vdup_n_u8(0x0F));
                uint8x8_t w1_u8 = vshr_n_u8(qs8, 4);

                int8x8_t w0_s8 = vsub_s8(vreinterpret_s8_u8(w0_u8), vdup_n_s8(8));
                int8x8_t w1_s8 = vsub_s8(vreinterpret_s8_u8(w1_u8), vdup_n_s8(8));

                int16x8_t w0_s16 = vmovl_s8(w0_s8);
                int16x8_t w1_s16 = vmovl_s8(w1_s8);

                int16x4_t w0_low = vget_low_s16(w0_s16);
                int16x4_t w0_high = vget_high_s16(w0_s16);
                int16x4_t w1_low = vget_low_s16(w1_s16);
                int16x4_t w1_high = vget_high_s16(w1_s16);

                float32x4_t w0_low_f32 = vcvtq_f32_s32(vmovl_s16(w0_low));
                float32x4_t w0_high_f32 = vcvtq_f32_s32(vmovl_s16(w0_high));
                float32x4_t w1_low_f32 = vcvtq_f32_s32(vmovl_s16(w1_low));
                float32x4_t w1_high_f32 = vcvtq_f32_s32(vmovl_s16(w1_high));

                float32x4_t x0_0 = vld1q_f32(x_block + k);
                float32x4_t x0_1 = vld1q_f32(x_block + k + 4);
                float32x4_t x1_0 = vld1q_f32(x_block + k + 16);
                float32x4_t x1_1 = vld1q_f32(x_block + k + 16 + 4);

                sumv = vmlaq_f32(sumv, w0_low_f32, x0_0);
                sumv = vmlaq_f32(sumv, w1_low_f32, x1_0);
                sumv = vmlaq_f32(sumv, w0_high_f32, x0_1);
                sumv = vmlaq_f32(sumv, w1_high_f32, x1_1);
            }

            float32x2_t sumv2 = vadd_f32(vget_low_f32(sumv), vget_high_f32(sumv));
            float v_block = vget_lane_f32(sumv2, 0) + vget_lane_f32(sumv2, 1);

            sum += v_block * d_scale;
        }
        output[i] = sum;
    }
#else
    for (NnSize i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnSize j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *b = &w[i * nBlocks + j];
            float v = 0.0f;
            for (NnSize k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                const int w0 = (b->qs[k] & 0x0F) - 8;
                const int w1 = (b->qs[k] >> 4) - 8;
                v += w0 * x[j * Q40_BLOCK_SIZE + k];
                v += w1 * x[j * Q40_BLOCK_SIZE + k + Q40_BLOCK_SIZE / 2];
            }
            sum += v * convertF16toF32(b->d);
        }
        output[i] = sum;
    }
#endif
}

static void matmul_Q80_Q40_F32(float *output, const NnBlockQ80 *x, const NnBlockQ40 *w, const NnSize n, const NnSize d, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    assert(n % Q40_BLOCK_SIZE == 0);
    const unsigned int nBlocks = n / Q40_BLOCK_SIZE;

#if defined(__ARM_NEON)
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    for (unsigned int di = start; di < end; di++) {
        float32x4_t sumv0 = vmovq_n_f32(0.0f);
        float32x4_t sumv1 = vmovq_n_f32(0.0f);
        float32x4_t sumv2 = vmovq_n_f32(0.0f);
        float32x4_t sumv3 = vmovq_n_f32(0.0f);

        unsigned int j = 0;
        
#if defined(__ARM_FEATURE_DOTPROD)
        for (; j + 3 < nBlocks; j += 4) {
            __builtin_prefetch(&w[di * nBlocks + j + 4]);
            __builtin_prefetch(&x[j + 4]);

            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ40 *w2 = &w[di * nBlocks + j + 2];
            const NnBlockQ40 *w3 = &w[di * nBlocks + j + 3];

            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];
            const NnBlockQ80 *x2 = &x[j + 2];
            const NnBlockQ80 *x3 = &x[j + 3];

            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w0->qs), m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w0->qs), 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w1->qs), m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w1->qs), 4)), s8b);
            int8x16_t w2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w2->qs), m4b)), s8b);
            int8x16_t w2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w2->qs), 4)), s8b);
            int8x16_t w3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w3->qs), m4b)), s8b);
            int8x16_t w3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w3->qs), 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);
            const int8x16_t x2l = vld1q_s8(x2->qs);
            const int8x16_t x2h = vld1q_s8(x2->qs + 16);
            const int8x16_t x3l = vld1q_s8(x3->qs);
            const int8x16_t x3h = vld1q_s8(x3->qs + 16);

            const int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0l, x0l), w0h, x0h);
            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1l, x1l), w1h, x1h);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w2l, x2l), w2h, x2h);
            const int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w3l, x3l), w3h, x3h);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), convertF16toF32(w0->d) * convertF16toF32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), convertF16toF32(w1->d) * convertF16toF32(x1->d));
            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), convertF16toF32(w2->d) * convertF16toF32(x2->d));
            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), convertF16toF32(w3->d) * convertF16toF32(x3->d));
        }
#else
        for (; j + 1 < nBlocks; j += 2) {
            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];

            const uint8x16_t w0qs = vld1q_u8(w0->qs);
            const uint8x16_t w1qs = vld1q_u8(w1->qs);
            
            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w0qs, m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w0qs, 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w1qs, m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w1qs, 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);

            const int16x8_t pl0l = vmull_s8(vget_low_s8(w0l), vget_low_s8(x0l));
            const int16x8_t pl0h = vmull_s8(vget_high_s8(w0l), vget_high_s8(x0l));
            const int16x8_t ph0l = vmull_s8(vget_low_s8(w0h), vget_low_s8(x0h));
            const int16x8_t ph0h = vmull_s8(vget_high_s8(w0h), vget_high_s8(x0h));
            
            const int16x8_t pl1l = vmull_s8(vget_low_s8(w1l), vget_low_s8(x1l));
            const int16x8_t pl1h = vmull_s8(vget_high_s8(w1l), vget_high_s8(x1l));
            const int16x8_t ph1l = vmull_s8(vget_low_s8(w1h), vget_low_s8(x1h));
            const int16x8_t ph1h = vmull_s8(vget_high_s8(w1h), vget_high_s8(x1h));

            const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), convertF16toF32(w0->d) * convertF16toF32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), convertF16toF32(w1->d) * convertF16toF32(x1->d));
        }
#endif

        for (; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[di * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];

            const uint8x16_t wqs = vld1q_u8(wb->qs);
            const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(wqs, m4b)), s8b);
            const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(wqs, 4)), s8b);

            const int8x16_t xl = vld1q_s8(xb->qs);
            const int8x16_t xh = vld1q_s8(xb->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl, xl), wh, xh);
#else
            const int16x8_t pll = vmull_s8(vget_low_s8(wl), vget_low_s8(xl));
            const int16x8_t plh = vmull_s8(vget_high_s8(wl), vget_high_s8(xl));
            const int16x8_t phl = vmull_s8(vget_low_s8(wh), vget_low_s8(xh));
            const int16x8_t phh = vmull_s8(vget_high_s8(wh), vget_high_s8(xh));
            
            const int32x4_t pl = vaddq_s32(vpaddlq_s16(pll), vpaddlq_s16(plh));
            const int32x4_t ph = vaddq_s32(vpaddlq_s16(phl), vpaddlq_s16(phh));
            const int32x4_t p = vaddq_s32(pl, ph);
#endif
            const float s = convertF16toF32(wb->d) * convertF16toF32(xb->d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), s);
        }

        output[di] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    }
#else
    for (NnSize i = start; i < end; i++) {
        float sum = 0.0;
        for (NnSize j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = convertF16toF32(wb->d) * convertF16toF32(xb->d);
            for (NnSize k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                const int w0 = (wb->qs[k] & 0x0F) - 8;
                const int w1 = (wb->qs[k] >> 4) - 8;
                const int i1 = xb->qs[k];
                const int i2 = xb->qs[k + Q40_BLOCK_SIZE / 2];
                sum += (w0 * i1 + w1 * i2) * s;
            }
        }
        output[i] = sum;
    }
#endif
}

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

static void gelu_F32(float *output, const unsigned int n, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

static void silu_F32(float *output, const unsigned int n, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int end4 = end - ((end - start) % 4);

    for (; i < end4; i += 4) {
        float32x4_t x = vld1q_f32(output + i);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t exp_negx = exp_F32_neon(neg_x);
        float32x4_t denominator = vaddq_f32(exp_negx, vdupq_n_f32(1.0f));

        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(recip, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(denominator, recip)));

        float32x4_t result = vmulq_f32(x, recip);
        vst1q_f32(output + i, result);
    }
#endif
    for (; i < end; i++) {
        float x = output[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

static void add_F32(float *output, const float *x, const unsigned int n, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        output[i] += x[i];
    }
}

static void add_Q80_F32(float *y, const NnBlockQ80 *x, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
    const NnSize nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

#if defined(__ARM_NEON)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = convertF16toF32(xi->d);
        float *y_base = y + i * Q80_BLOCK_SIZE;
        const int8_t *qs = xi->qs;

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 16) {
            // Load 16x 8-bit quantized values
            const int8x16_t q8 = vld1q_s8(qs + j);
            
            // Split into 8-bit high/low components
            const int8x8_t q8_low = vget_low_s8(q8);
            const int8x8_t q8_high = vget_high_s8(q8);
            
            // Sign extend to 16-bit
            const int16x8_t q16_low = vmovl_s8(q8_low);
            const int16x8_t q16_high = vmovl_s8(q8_high);
            
            // Sign extend to 32-bit and convert to float
            const float32x4_t qf_ll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_low)));
            const float32x4_t qf_lh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_low)));
            const float32x4_t qf_hl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_high)));
            const float32x4_t qf_hh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_high)));
            
            // Multiply by scale factor
            const float32x4_t sf_ll = vmulq_n_f32(qf_ll, xid);
            const float32x4_t sf_lh = vmulq_n_f32(qf_lh, xid);
            const float32x4_t sf_hl = vmulq_n_f32(qf_hl, xid);
            const float32x4_t sf_hh = vmulq_n_f32(qf_hh, xid);
            
            // Load existing y values
            float32x4_t y_ll = vld1q_f32(y_base + j);
            float32x4_t y_lh = vld1q_f32(y_base + j + 4);
            float32x4_t y_hl = vld1q_f32(y_base + j + 8);
            float32x4_t y_hh = vld1q_f32(y_base + j + 12);
            
            // Accumulate results
            y_ll = vaddq_f32(y_ll, sf_ll);
            y_lh = vaddq_f32(y_lh, sf_lh);
            y_hl = vaddq_f32(y_hl, sf_hl);
            y_hh = vaddq_f32(y_hh, sf_hh);
            
            // Store results back
            vst1q_f32(y_base + j, y_ll);
            vst1q_f32(y_base + j + 4, y_lh);
            vst1q_f32(y_base + j + 8, y_hl);
            vst1q_f32(y_base + j + 12, y_hh);
        }
    }
#else
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = convertF16toF32(xi->d);
        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j++) {
            y[i * Q80_BLOCK_SIZE + j] += xid * xi->qs[j];
        }
    }
#endif
}

static void softmax_F32(float *x, const NnSize size) {
    if (size == 0)
        return;

#if defined(__ARM_NEON)
    NnSize j;
    float maxVal;
    if (size >= 4) {
        float32x4_t fs;
        float32x4_t fmaxv = vld1q_f32(&x[0]);
        j = size - (size % 4);
        for (NnSize i = 4; i < j; i += 4) {
            fs = vld1q_f32(&x[i]);
            fmaxv = vmaxq_f32(fmaxv, fs);
        }
        maxVal = vmaxvq_f32(fmaxv);
    } else {
        maxVal = x[0];
        j = 1;
    }
    for (; j < size; j++)
        maxVal = fmaxf(maxVal, x[j]);

    const float32x4_t maxVal_vec = vdupq_n_f32(maxVal);
    float32x4_t sumv = vdupq_n_f32(0.0f);
    NnSize i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vsubq_f32(val, maxVal_vec);
        val = exp_F32_neon(val);
        vst1q_f32(x + i, val);
        sumv = vaddq_f32(sumv, val);
    }

    float32x2_t sum_lo = vadd_f32(vget_low_f32(sumv), vget_high_f32(sumv));
    float sum = vget_lane_f32(sum_lo, 0) + vget_lane_f32(sum_lo, 1);

    for (; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0f)
        sum = 0.000001f;

    const float inv_sum = 1.0f / sum;
    const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

    i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vmulq_f32(val, inv_sum_vec);
        vst1q_f32(x + i, val);
    }
    for (; i < size; ++i)
        x[i] /= sum;
#else
    float maxVal = x[0];
    for (NnSize i = 1; i < size; i++) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }
    float sum = 0.0f;
    for (NnSize i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    if (sum == 0.0)
        sum = 0.000001;
    for (NnSize i = 0; i < size; i++)
        x[i] /= sum;
#endif
}

static float dotProduct_F32(const float *a, const float *b, const unsigned int size) {
    #if defined(__ARM_NEON)
        assert(size % 4 == 0);
        float32x4_t fa;
        float32x4_t fb;
        float32x4_t fs = vmovq_n_f32(0);
        for (unsigned int i = 0; i < size; i += 4) {
            fa = vld1q_f32(&a[i]);
            fb = vld1q_f32(&b[i]);
            fs = vmlaq_f32(fs, fa, fb);
        }
        return vaddvq_f32(fs);
    #elif defined(__AVX2__)
        assert(size % 8 == 0);
        __m256 a0, b0;
        __m256 u = _mm256_set1_ps(0.0f);
        for (unsigned int i = 0; i < size; i += 8) {
            a0 = _mm256_loadu_ps(&a[i]);
            b0 = _mm256_loadu_ps(&b[i]);
            u = _mm256_fmadd_ps(a0, b0, u);
        }
        return hsum_F8(u);
    #else
        float sum = 0.0f;
        for (unsigned int i = 0; i < size; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    #endif
}

static void multiheadAtt_F32(
    float *x, const float *q, float *att, float *keyCache, float *valueCache,
    const unsigned pos, const NnSize nHeads, const NnSize nHeads0, const NnSize nKvHeads, const NnSize kvDim0, const NnSize headSize, const NnSize seqLen,
    const NnSize nThreads, const NnSize threadIndex) 
{
    SPLIT_THREADS(h0Start, h0End, nHeads0, nThreads, threadIndex);
    const NnSize kvMul = nHeads / nKvHeads;
    const float headSizeRoot = sqrtf(headSize);

    for (NnSize h0 = h0Start; h0 < h0End; h0++) {
        const float *hQ = &q[h0 * headSize];
        const NnSize headIndex = h0 / kvMul;
        const float *hKc = &keyCache[headIndex * headSize];
        const float *hVc = &valueCache[headIndex * headSize];
        float *hAtt = &att[h0 * seqLen];

        for (NnSize t = 0; t <= pos; t++) {
            const float *posK = &hKc[t * kvDim0];
            const float score = dotProduct_F32(hQ, posK, headSize) / headSizeRoot;
            hAtt[t] = score;
        }

        softmax_F32(hAtt, pos + 1);

        float *hX = &x[h0 * headSize];
        std::memset(hX, 0, headSize * sizeof(float));

        for (NnSize t = 0; t <= pos; t++) {
            const float *posV = &hVc[t * kvDim0];
            const float posA = hAtt[t];
            for (int i = 0; i < headSize; i++) {
                hX[i] += posA * posV[i];
            }
        }
    }
}

static void mul_F32(float *output, const float *x, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
    assert(n % 4 == 0);
    NnSize nBlocks = n / 4;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

#if defined(__ARM_NEON)
    assert(n % 16 == 0);
    const float *xi = x + start * 4;
    float *oi = output + start * 4;
    NnSize blocksRemaining = end - start;

    while (blocksRemaining >= 4) {
        float32x4_t x0 = vld1q_f32(xi);
        float32x4_t o0 = vld1q_f32(oi);
        o0 = vmulq_f32(o0, x0);
        vst1q_f32(oi, o0);
        xi += 4; oi += 4;

        x0 = vld1q_f32(xi);
        o0 = vld1q_f32(oi);
        vst1q_f32(oi, vmulq_f32(o0, x0));
        xi += 4; oi += 4;

        x0 = vld1q_f32(xi);
        o0 = vld1q_f32(oi);
        vst1q_f32(oi, vmulq_f32(o0, x0));
        xi += 4; oi += 4;

        x0 = vld1q_f32(xi);
        o0 = vld1q_f32(oi);
        vst1q_f32(oi, vmulq_f32(o0, x0));
        xi += 4; oi += 4;

        blocksRemaining -= 4;
    }

    while (blocksRemaining-- > 0) {
        float32x4_t x_vec = vld1q_f32(xi);
        float32x4_t o_vec = vld1q_f32(oi);
        vst1q_f32(oi, vmulq_f32(o_vec, x_vec));
        xi += 4; oi += 4;
    }
#else
    for (NnSize i = start; i < end; i++) {
        const float *xi = &x[i * 4];
        float *oi = &output[i * 4];
        oi[0] *= xi[0];
        oi[1] *= xi[1];
        oi[2] *= xi[2];
        oi[3] *= xi[3];
    }
#endif
}

static void mul_Q80_F32(float *output, const NnBlockQ80 *x, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
    const NnSize nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);
    for (NnSize i = start; i < end; i++) {
        const NnBlockQ80 *b = &x[i];
        float d = convertF16toF32(b->d);
        for (NnSize j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnSize k = i * Q80_BLOCK_SIZE + j;
            output[k] *= d * b->qs[j];
        }
    }
}

static void copy_UNK(NnByte *output, const NnByte *x, NnSize size, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    NnSize s = end - start;
    if (s != 0)
        std::memcpy(&output[start], &x[start], s);
}

//

static void mergeAddForward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    NnSize nSlices = context->inputSize.x / context->outputSize.x;

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        float *input = (float *)context->input[batchIndex];
        for (NnSize sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            float *i = &input[sliceIndex * context->outputSize.x];
            DEBUG_VECTOR(context, "input", i);
            add_F32(
                output,
                i,
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void mergeAddForward_Q80_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    assert(context->inputSize.floatType == F_Q80);
    assert(context->outputSize.floatType == F_32);

    NnSize nSlices = context->inputSize.x / context->outputSize.x;
    NnSize xSize = context->outputSize.x / Q80_BLOCK_SIZE;
    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        for (NnSize sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            add_Q80_F32(
                output,
                &input[sliceIndex * xSize],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void embeddingForward_F32_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, 1);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.x, context->outputSize.x);

    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnSize token = (NnSize)*((float *)context->input[batchIndex]);
        copy_UNK(
            context->output[batchIndex],
            &context->weight[token * dimSize],
            dimSize,
            nThreads,
            threadIndex);
    }
}

static void embeddingForward_F32_F32_Q80(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnSize token = (NnSize)*((float *)context->input[batchIndex]);
        quantizeF32toQ80(
            (float *)&context->weight[token * dimSize],
            (NnBlockQ80 *)context->output[batchIndex],
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void rmsForward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    if (threadIndex == 0) {
        ASSERT_EQ(context->inputSize.y, context->nBatches);
        ASSERT_EQ(context->outputSize.x, 1);
        ASSERT_EQ(context->outputSize.y, context->nBatches);

        const NnRmsOpConfig *config = (NnRmsOpConfig *)context->opConfig;
        for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            float *input = (float *)context->input[batchIndex];
            float *output = (float *)context->output[batchIndex];
            DEBUG_VECTOR(context, "input", input);
            float rms = rms_F32(
                input,
                context->inputSize.x,
                config->epsilon);
            output[0] = rms;
            DEBUG_SCALAR(context, "output", rms);
        }
    }
}

static void rmsNormForward_F32_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;

    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.floatType, F_32);
    ASSERT_EQ(context->weightSize.y, 1);
    ASSERT_EQ(context->weightSize.x, context->inputSize.x);
    NnBufferConfig *rmsBufferConfig = &context->bufferConfigs[config->rmsBufferIndex];
    ASSERT_EQ(rmsBufferConfig->size.floatType, F_32);
    ASSERT_EQ(rmsBufferConfig->size.x, 1);
    ASSERT_EQ(rmsBufferConfig->size.y, context->nBatches);

    const float *weight = (float *)context->weight;
    const float *rms = (float *)context->buffers[config->rmsBufferIndex];

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_F32(
            output,
            input,
            rms[batchIndex],
            weight,
            context->inputSize.x,
            nThreads,
            threadIndex);
    }
}

static void rmsNormForward_Q80_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    const float *weight = (float *)context->weight;
    const float *rms = (float *)context->buffers[config->rmsBufferIndex];

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_Q80_F32_F32(
            output,
            input,
            rms[batchIndex],
            weight,
            context->inputSize.x,
            nThreads,
            threadIndex);
        DEBUG_VECTOR(context, "output", output);
    }
}

static void initMatmulForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->weightSize.y);
    ASSERT_EQ(context->outputSize.x, context->weightSize.x);

    if (!context->hasInputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for input\n", context->name);
    if (!context->hasOutputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for output\n", context->name);

}

static bool matmulForward_llamafile(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    if (batchSize == 1 || !context->hasInputContinuousMemory || !context->hasOutputContinuousMemory)
        return false;

    const NnSize n = context->weightSize.y / getBlockSize(context->inputSize.floatType);
    const NnSize d = context->weightSize.x;
    return llamafile_sgemm(
        d, batchSize, n,
        context->weight, n,
        context->input[0], n,
        context->output[0], d,
        threadIndex, nThreads, 0,
        context->weightSize.floatType,
        context->inputSize.floatType,
        F_32
    );
}

static void matmulForward_F32_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const float *weight = (float *)context->weight;
    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        matmul_F32_F32_F32(
            output,
            input,
            weight,
            context->weightSize.y,
            context->weightSize.x,
            nThreads,
            threadIndex);
    }
}

static void matmulForward_F32_Q40_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    const NnBlockQ40 *weight = (NnBlockQ40 *)context->weight;
    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        matmul_F32_Q40_F32(
            output,
            input,
            weight,
            context->weightSize.y,
            context->weightSize.x,
            nThreads,
            threadIndex);
        DEBUG_VECTOR(context, "output", output);
    }
}

static void matmulForward_Q80_Q40_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const NnBlockQ40 *weight = (NnBlockQ40 *)context->weight;
    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        matmul_Q80_Q40_F32(
            output,
            input,
            weight,
            context->weightSize.y,
            context->weightSize.x,
            nThreads,
            threadIndex);
    }
}

static void siluForward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->weightSize.nBytes, 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        silu_F32(output, context->outputSize.x, nThreads, threadIndex);
    }
}

static void geluForward_F32_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->weightSize.nBytes, 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        gelu_F32(output, context->outputSize.x, nThreads, threadIndex);
    }
}

static void initRopeLlama31Forward(NnCpuOpContext *context) {
    const NnRopeLlama31OpConfig *config = (NnRopeLlama31OpConfig *)context->opConfig;
    if (context->bufferFlags[config->ropeCacheBufferIndex] == 1)
        return;
    context->bufferFlags[config->ropeCacheBufferIndex] = 1;

    const NnRopeSlice *slice = &config->slice;
    float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];

    for (NnSize pos = 0; pos < slice->seqLen; pos++) {
        for (NnSize i = slice->kvDimStart; i < slice->qDimEnd; i += 2) {
            const NnSize headDim = i % slice->headSize;
            const float freq = 1.0f / powf(slice->ropeTheta, headDim / (float)slice->headSize);
            const float val = pos * freq;
            const float fcr = cosf(val);
            const float fci = sinf(val);
            cache[pos * slice->sliceDim + (i - slice->kvDimStart)] = fcr;
            cache[pos * slice->sliceDim + (i - slice->kvDimStart) + 1] = fci;
        }
    }
}

static inline float ropeLlama31Scale(const float freq, const NnRopeLlama31OpConfig *config) {
    const float waveLen = 2.0f * M_PI * freq;
    const float highFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingHighFreqFactory;
    if (waveLen < highFreqWavelen) {
        return freq;
    }
    const float lowFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingLowFreqFactor;
    if (waveLen > lowFreqWavelen) {
        return freq / config->ropeScalingFactor;
    }
    const float smooth = (config->ropeScalingOrigMaxSeqLen / waveLen - config->ropeScalingLowFreqFactor) / (config->ropeScalingHighFreqFactory - config->ropeScalingLowFreqFactor);
    return (1 - smooth) * freq / config->ropeScalingFactor + smooth * freq;
}

static void ropeLlama31Forward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    const NnRopeLlama31OpConfig *config = (NnRopeLlama31OpConfig *)context->opConfig;
    const NnRopeSlice *slice = &config->slice;
    const float *positions = (float *)context->pipes[config->positionPipeIndex];
    const float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];

    const NnSize dim0Half = (config->isQ ? slice->qDim0 : slice->kvDim0) / 2;
    const NnSize shift = config->isQ ? slice->qShift : 0;
    SPLIT_THREADS(s, e, dim0Half, nThreads, threadIndex);
    const NnSize iStart = s * 2;
    const NnSize iEnd = e * 2;

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *x = (float *)context->input[batchIndex];
        const NnSize pos = (NnSize)positions[batchIndex];
        const float *c = &cache[pos * slice->sliceDim + shift];

        for (NnSize i = iStart; i < iEnd; i += 2) {
            const float fcr = c[i];
            const float fci = c[i + 1];
            const float v0 = x[i];
            const float v1 = x[i + 1];
            x[i] = ropeLlama31Scale(v0 * fcr - v1 * fci, config);
            x[i + 1] = ropeLlama31Scale(v0 * fci + v1 * fcr, config);
        }
    }
}

static void multiHeadAttForward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    ASSERT_EQ(context->weightSize.nBytes, 0);
    ASSERT_EQ(context->inputSize.x, config->qSlice.d0);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    NnSize2D *querySize = &context->bufferConfigs[config->queryBufferIndex].size;
    ASSERT_EQ(querySize->x, config->qSlice.d0);
    NnSize2D *posSize = &context->pipeConfigs[config->positionPipeIndex].size;
    ASSERT_EQ(posSize->x, 1);
    ASSERT_EQ(posSize->y, context->nBatches);

    float *query = (float *)context->buffers[config->queryBufferIndex];
    float *keyCache = (float *)context->buffers[config->keyCacheBufferIndex];
    float *valueCache = (float *)context->buffers[config->valueCacheBufferIndex];
    float *att = (float *)context->buffers[config->attBufferIndex];
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *i = (float *)context->input[batchIndex];
        float *q = &query[batchIndex * config->qSlice.d0];
        NnSize pos = (NnSize)positions[batchIndex];
        assert(pos < config->seqLen);

        DEBUG_VECTOR(context, "input", i);
        DEBUG_VECTOR(context, "q", q);

        multiheadAtt_F32(i, q, att, keyCache, valueCache, pos,
            config->multiHeadAttSlice.nHeads, config->multiHeadAttSlice.nHeads0,
            config->nKvHeads, config->kvCacheSlice.kvDim0, config->headSize, config->seqLen, nThreads, threadIndex);
    }
}

static void mulForward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->weightSize.nBytes, 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        mul_F32(
            output,
            input,
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void mulForward_Q80_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        mul_Q80_F32(
            output,
            input,
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void initCastForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
}

static void castForward_UNK(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    const NnSize rowBytes = context->outputSize.nBytes / context->outputSize.y;

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        copy_UNK(
            context->output[batchIndex],
            context->input[batchIndex],
            rowBytes,
            nThreads,
            threadIndex);
    }
}

static void castForward_F32_Q80(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        NnBlockQ80 *output = (NnBlockQ80 *)context->output[batchIndex];
        quantizeF32toQ80(
            input,
            output,
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void castForward_Q80_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->outputSize.floatType, F_32);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        dequantizeQ80toF32(
            input,
            output,
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

// device

void printCpuInstructionSet() {
    printf("🧠 %16s:", "CPU");
#if defined(__ARM_NEON)
    printf(" neon");
#if defined(__ARM_FEATURE_DOTPROD)
    printf(" dotprod");
#endif
#if defined(__ARM_FP16_FORMAT_IEEE)
    printf(" fp16");
#endif
#endif
#if defined(__AVX2__)
    printf(" avx2");
#endif
    printf("\n");
}

NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_ROPE_LLAMA_3_1)
        return initRopeLlama31Forward;
    if (code == OP_MATMUL)
        return initMatmulForward;
    if (code == OP_CAST)
        return initCastForward;
    return nullptr;
}

NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_MERGE_ADD) {
        if (quantType == F32_F32_F32) return mergeAddForward_F32_F32;
        if (quantType == Q80_Q80_F32) return mergeAddForward_Q80_F32;
    }
    if (code == OP_EMBEDDING) {
        if (quantType == F32_F32_F32) return embeddingForward_F32_F32_F32;
        if (quantType == F32_F32_Q80) return embeddingForward_F32_F32_Q80;
    }
    if (code == OP_RMS) {
        if (quantType == F32_F32_F32) return rmsForward_F32_F32;
    }
    if (code == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return rmsNormForward_F32_F32_F32;
        if (quantType == Q80_F32_F32) return rmsNormForward_Q80_F32_F32;
    }
    if (code == OP_MATMUL) {
        if (quantType == F32_F32_F32) return matmulForward_F32_F32_F32;
        if (quantType == F32_Q40_F32) return matmulForward_F32_Q40_F32;
        if (quantType == Q80_Q40_F32) return matmulForward_Q80_Q40_F32;
    }
    if (code == OP_ROPE_LLAMA_3_1) {
        if (quantType == F32_F32_F32) return ropeLlama31Forward_F32_F32;
    }
    if (code == OP_MULTIHEAD_ATT) {
        if (quantType == F32_F32_F32) return multiHeadAttForward_F32_F32;
    }
    if (code == OP_GELU) {
        if (quantType == F32_F32_F32) return geluForward_F32_F32_F32;
    }
    if (code == OP_SILU) {
        if (quantType == F32_F32_F32) return siluForward_F32_F32;
    }
    if (code == OP_MUL) {
        if (quantType == F32_F32_F32) return mulForward_F32_F32;
        if (quantType == Q80_Q80_F32) return mulForward_Q80_F32;
    }
    if (code == OP_CAST) {
        if (quantType == F32_F32_F32) return castForward_UNK;
        if (quantType == F32_F32_Q80) return castForward_F32_Q80;
        if (quantType == Q80_Q80_Q80) return castForward_UNK;
        if (quantType == Q80_Q80_F32) return castForward_Q80_F32;
    }
    return nullptr;
}
