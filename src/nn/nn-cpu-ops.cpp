#include "nn-cpu-ops.hpp"
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#ifdef _WIN32
    #define _USE_MATH_DEFINES
#endif
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#endif

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

#define SPLIT_THREADS(varStart, varEnd, rangeLen, nThreads, threadIndex) \
    const NnSize rangeSlice = rangeLen / nThreads; \
    const NnSize rangeRest = rangeLen % nThreads; \
    const NnSize varStart = threadIndex * rangeSlice + (threadIndex < rangeRest ? threadIndex : rangeRest); \
    const NnSize varEnd = varStart + rangeSlice + (threadIndex < rangeRest ? 1 : 0);

#if defined(__AVX2__)
    static inline float hsum_float_8(const __m256 x) {
        __m128 res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        return _mm_cvtss_f32(res);
    }
#endif

static float convertF16toF32(const uint16_t value) {
    union Fl32 {
        uint32_t u;
        float f;
    };
    const Fl32 magic = { (254U - 15U) << 23 };
    const Fl32 infNan = { (127U + 16U) << 23 };
    Fl32 result;
    result.u = (value & 0x7FFFU) << 13;
    result.f *= magic.f;
    if (result.f >= infNan.f)
        result.u |= 255U << 23;
    result.u |= (value & 0x8000U) << 16;
    return result.f;
}

static uint16_t convertF32ToF16(const float x) {
#if defined(__ARM_NEON)
    __fp16 h = x;
    return *(uint16_t *)&h;
#else
    int i = *(int *)&x;
    int s = (i >> 16) & 0x00008000;
    int e = ((i >> 23) & 0x000000ff) - (127 - 15);
    int m = i & 0x007fffff;
    if (e <= 0) {
        if (e < -10) {
            return s;
        }
        m = m | 0x00800000;
        int t = 14 - e;
        int a = (1 << (t - 1)) - 1;
        int b = (m >> t) & 1;
        m = (m + a + b) >> t;
        return s | m;
    }
    if (e == 0xff - (127 - 15)) {
        if (m == 0) {
            return s | 0x7c00;
        }
        m >>= 13;
        return s | 0x7c00 | m | (m == 0);
    }
    m = m + 0x00000fff + ((m >> 13) & 1);
    if (m & 0x00800000) {
        m =  0;
        e += 1;
    }
    assert(e <= 30);
    return s | (e << 10) | (m >> 13);
#endif
}

static void quantizeF32toQ80(const float *input, NnBlockQ80 *output, const NnSize k, const NnSize nThreads, const NnSize threadIndex) {
    assert(k % Q80_BLOCK_SIZE == 0);

    const int nBlocks = k / Q80_BLOCK_SIZE;
    const int blocksPerThread = nBlocks / nThreads;
    const int sk = blocksPerThread * Q80_BLOCK_SIZE;
    const int currentThreadBlocks = blocksPerThread + (threadIndex == nThreads - 1 ? nBlocks % nThreads : 0);

    const float *x = &input[sk * threadIndex];
    NnBlockQ80 *y = &output[blocksPerThread * threadIndex];

#if defined(__ARM_NEON)
    float dBuf[4];

    for (int i = 0; i < currentThreadBlocks; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
        for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
        for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        int dbi = i % 4;
        dBuf[dbi] = d;
        if (dbi == 3) {
            float32x4_t dBuf32 = vld1q_f32(dBuf);
            int16x4_t dBuf16 = (int16x4_t)vcvt_f16_f32(dBuf32);

            y[i - 3].d = dBuf16[0];
            y[i - 2].d = dBuf16[1];
            y[i - 1].d = dBuf16[2];
            y[i - 0].d = dBuf16[3];
        }

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
        }
    }

    int rest = currentThreadBlocks % 4;
    if (rest != 0) {
        float32x4_t dBuf32 = vld1q_f32(dBuf);
        int16x4_t dBuf16 = (int16x4_t)vcvt_f16_f32(dBuf32);
        for (int i = 0; i < rest; i++) {
            y[currentThreadBlocks - rest + i].d = dBuf16[i];
        }
    }
#else
    for (int i = 0; i < currentThreadBlocks; i++) {
        float amax = 0.0f;

        for (int j = 0; j < Q80_BLOCK_SIZE; j++) {
            const float v = fabsf(x[i * Q80_BLOCK_SIZE + j]);
            amax = amax > v ? amax : v;
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = convertF32ToF16(d);

        for (int j = 0; j < Q80_BLOCK_SIZE; ++j) {
            const float x0 = x[i * Q80_BLOCK_SIZE + j] * id;
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

static void dequantizeQ80toF32(const NnBlockQ80 *input, float* output, const NnSize k, const NnSize nThreads, const NnSize threadIndex) {
    assert(k % Q80_BLOCK_SIZE == 0);
    const int nBlocks = k / Q80_BLOCK_SIZE;
    const int blocksPerThread = nBlocks / nThreads;
    const int sk = blocksPerThread * Q80_BLOCK_SIZE;
    const int currentThreadBlocks = blocksPerThread + (threadIndex == nThreads - 1 ? nBlocks % nThreads : 0);

    const NnBlockQ80 *x = &input[blocksPerThread * threadIndex];
    float* y = &output[sk * threadIndex];

    for (int i = 0; i < currentThreadBlocks; i++) {
        const float d = convertF16toF32(x[i].d);
        for (int j = 0; j < Q80_BLOCK_SIZE; j++) {
            y[i * Q80_BLOCK_SIZE + j] = x[i].qs[j] * d;
        }
    }
}

static void quantizeF32toQ40(const float *x, NnBlockQ40 *output, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
    assert(n % Q40_BLOCK_SIZE == 0);
    const NnSize nBlocks = n / Q40_BLOCK_SIZE;
    const NnSize halfSize = Q40_BLOCK_SIZE / 2;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

    for (NnSize i = start; i < end; i++) {
        float amax = 0.0f;
        float max = 0.0f;
        for (NnSize j = 0; j < Q40_BLOCK_SIZE; j++) {
            float v = x[i * Q40_BLOCK_SIZE + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;

        NnBlockQ40 *o = &output[i];
        o->d = convertF32ToF16(d);
        for (NnSize j = 0; j < halfSize; j++) {
            const float x0 = x[i * Q40_BLOCK_SIZE + j] * id;
            const float x1 = x[i * Q40_BLOCK_SIZE + halfSize + j] * id;
    
            uint8_t xi0 = (int8_t)(x0 + 8.5f);
            uint8_t xi1 = (int8_t)(x1 + 8.5f);
            if (xi0 > 15) xi0 = 15;
            if (xi1 > 15) xi1 = 15;

            o->qs[j] = xi0 | (xi1 << 4);
        }
    }
}

static void dequantizeQ40toF32(const NnBlockQ40 *x, float *output, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
    assert(n % Q40_BLOCK_SIZE == 0);
    const NnSize nBlocks = n / Q40_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

    for (NnSize i = start; i < end; i++) {
        const NnBlockQ40 *b = &x[i];
        const float d = convertF16toF32(b->d);

        for (int j = 0; j < Q40_BLOCK_SIZE / 2; ++j) {
            const int x0 = (b->qs[j] & 0x0F) - 8;
            const int x1 = (b->qs[j] >> 4) - 8;

            output[i * Q40_BLOCK_SIZE + j] = x0 * d;
            output[i * Q40_BLOCK_SIZE + j + Q40_BLOCK_SIZE / 2] = x1 * d;
        }
    }
}

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
    ss = hsum_float_8(u);
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
        output[i] = hsum_float_8(u);
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
    const NnBlockQ80* input = x;
    assert(n % Q40_BLOCK_SIZE == 0);
    const unsigned int nBlocks = n / Q40_BLOCK_SIZE;

#if defined(__ARM_NEON)
    float32x4_t sumv0;
    float32x4_t sumv1;
    for (unsigned int d = start; d < end; d++) {
        sumv0 = vmovq_n_f32(0);
        sumv1 = vmovq_n_f32(0);
        for (unsigned int j = 0; j < nBlocks; j += 2) {
            const NnBlockQ40 *x0 = &w[d * nBlocks + j];
            const NnBlockQ40 *x1 = &w[d * nBlocks + j + 1];
            const NnBlockQ80 *y0 = &input[j];
            const NnBlockQ80 *y1 = &input[j + 1];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
            const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
            const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
            const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t v1_0l = vld1q_s8(y0->qs);
            const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
            const int8x16_t v1_1l = vld1q_s8(y1->qs);
            const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
            const int32x4_t p_1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), convertF16toF32(x0->d) * convertF16toF32(y0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), convertF16toF32(x1->d) * convertF16toF32(y1->d));
#else
            const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0l));
            const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
            const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0h));
            const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

            const int16x8_t pl1l = vmull_s8(vget_low_s8 (v0_1ls), vget_low_s8 (v1_1l));
            const int16x8_t pl1h = vmull_s8(vget_high_s8(v0_1ls), vget_high_s8(v1_1l));
            const int16x8_t ph1l = vmull_s8(vget_low_s8 (v0_1hs), vget_low_s8 (v1_1h));
            const int16x8_t ph1h = vmull_s8(vget_high_s8(v0_1hs), vget_high_s8(v1_1h));

            const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), convertF16toF32(x0->d) * convertF16toF32(y0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), convertF16toF32(x1->d) * convertF16toF32(y1->d));
#endif
        }
        output[d] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    }
#else
    assert(false); // TOOD
#endif
}

static void matmul_F32_Q40_Q80(NnBlockQ80 *output, const float *x, const NnBlockQ40 *w, const NnSize n, const NnSize d, const NnSize nThreads, const NnSize threadIndex) {
    assert(n % Q40_BLOCK_SIZE == 0);
    assert(d % Q80_BLOCK_SIZE == 0);
    const NnSize nBlocks = n / Q40_BLOCK_SIZE;
    const NnSize dBlocks = d / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, dBlocks, nThreads, threadIndex);

    float temp[Q80_BLOCK_SIZE] __attribute__((aligned(16)));
#if defined(__ARM_NEON)
    for (NnSize i = start; i < end; i++) {
        NnBlockQ80 *o = &output[i];
        float32x4_t v_amax = vdupq_n_f32(0.0f);

        for (NnSize j = 0; j < Q80_BLOCK_SIZE; j++) {
            float sum = 0.0f;
            for (NnSize k = 0; k < nBlocks; k++) {
                const NnBlockQ40 *b = &w[(i * Q80_BLOCK_SIZE + j) * nBlocks + k];
                const float scale = convertF16toF32(b->d);
                float sum_v = 0.0f;

                for (NnSize l_start = 0; l_start < Q40_BLOCK_SIZE / 2; l_start += 8) {
                    uint8x8_t qs = vld1_u8(b->qs + l_start);

                    int8x8_t w0 = vsub_s8(vreinterpret_s8_u8(vand_u8(qs, vdup_n_u8(0x0F))), vdup_n_s8(8));
                    int8x8_t w1 = vsub_s8(vreinterpret_s8_u8(vshr_n_u8(qs, 4)), vdup_n_s8(8));

                    int16x8_t w0_16 = vmovl_s8(w0);
                    float32x4_t w0_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_16)));
                    float32x4_t w0_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_16)));

                    int16x8_t w1_16 = vmovl_s8(w1);
                    float32x4_t w1_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1_16)));
                    float32x4_t w1_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1_16)));

                    const float* x_base = x + k * Q40_BLOCK_SIZE + l_start;
                    float32x4_t x0 = vld1q_f32(x_base);
                    float32x4_t x1 = vld1q_f32(x_base + 4);
                    float32x4_t x0w = vld1q_f32(x_base + 16);
                    float32x4_t x1w = vld1q_f32(x_base + 20);

                    float32x4_t p0 = vmulq_f32(w0_l, x0);
                    p0 = vmlaq_f32(p0, w1_l, x0w);
                    float32x4_t p1 = vmulq_f32(w0_h, x1);
                    p1 = vmlaq_f32(p1, w1_h, x1w);

                    sum_v += vaddvq_f32(p0) + vaddvq_f32(p1);
                }
                sum += sum_v * scale;
            }
            temp[j] = sum;
        }

        for (int j = 0; j < Q80_BLOCK_SIZE; j += 4) {
            float32x4_t vec = vabsq_f32(vld1q_f32(temp + j));
            v_amax = vmaxq_f32(v_amax, vec);
        }
        float amax = vmaxvq_f32(v_amax);

        const float oD = amax / 127.0f;
        const float id = oD != 0.0f ? 1.0f / oD : 0.0f;
        o->d = convertF32ToF16(oD);
        
        const float32x4_t vid = vdupq_n_f32(id);
        for (int j = 0; j < Q80_BLOCK_SIZE; j += 8) {
            float32x4_t v0 = vmulq_f32(vld1q_f32(temp + j), vid);
            float32x4_t v1 = vmulq_f32(vld1q_f32(temp + j + 4), vid);
            
            int32x4_t i0 = vcvtnq_s32_f32(v0);
            int32x4_t i1 = vcvtnq_s32_f32(v1);
            
            int16x4_t s0 = vqmovn_s32(i0);
            int16x4_t s1 = vqmovn_s32(i1);
            int8x8_t res = vqmovn_s16(vcombine_s16(s0, s1));
            
            vst1_s8(o->qs + j, res);
        }
    }
#else
    for (NnSize i = start; i < end; i++) {
        NnBlockQ80 *o = &output[i];
        float amax = 0.0f;

        for (NnSize j = 0; j < Q80_BLOCK_SIZE; j++) {
            float sum = 0.0f;
            for (NnSize k = 0; k < nBlocks; k++) {
                const NnBlockQ40 *b = &w[(i * Q80_BLOCK_SIZE + j) * nBlocks + k];
                float v = 0.0f;
                for (NnSize l = 0; l < Q40_BLOCK_SIZE / 2; l++) {
                    const int w0 = (b->qs[l] & 0x0F) - 8;
                    const int w1 = (b->qs[l] >> 4) - 8;
                    v += w0 * x[k * Q40_BLOCK_SIZE + l];
                    v += w1 * x[k * Q40_BLOCK_SIZE + l + Q40_BLOCK_SIZE / 2];
                }
                sum += v * convertF16toF32(b->d);
            }
            const float sumAbs = fabsf(sum);
            amax = amax > sumAbs ? amax : sumAbs;
            temp[j] = sum;
        }

        const float oD = amax / ((1 << 7) - 1);
        const float id = oD ? 1.0f / oD : 0.0f;
        o->d = convertF32ToF16(oD);
        for (int j = 0; j < Q80_BLOCK_SIZE; j++)
            o->qs[j] = roundf(temp[j] * id);
    }
#endif
}

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

static void geluF32(float *output, const unsigned int n, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

static void siluF32(float *output, const unsigned int n, const NnSize nThreads, const NnSize threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
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

static void softmaxF32(float *x, const unsigned int size) {
    if (size == 0)
        return;
    float maxVal;
#if defined(__ARM_NEON)
    unsigned int j;
    if (size >= 4) {
        float32x4_t fs;
        float32x4_t fmaxv = vld1q_f32(&x[0]);
        j = size - (size % 4);
        for (unsigned int i = 4; i < j; i += 4) {
            fs = vld1q_f32(&x[i]);
            fmaxv = vmaxq_f32(fmaxv, fs);
        }
        maxVal = vmaxvq_f32(fmaxv);
    } else {
        maxVal = x[0];
        j = 1;
    }
    for (unsigned int i = j; i < size; i++) {
        maxVal = fmaxf(maxVal, x[i]);
    }
#else
    // find max value (for numerical stability)
    maxVal = x[0];
    for (unsigned int i = 1; i < size; i++) {
        if (x[i] > maxVal) {
            maxVal = x[i];
        }
    }
#endif
    // exp and sum
    float sum = 0.0f;
    for (unsigned int i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    if (sum == 0.0) sum = 0.000001;
    for (unsigned int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static float dotProductF32(const float *a, const float *b, const unsigned int size) {
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
        return hsum_float_8(u);
    #else
        float sum = 0.0f;
        for (unsigned int i = 0; i < size; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    #endif
}

static void multiheadAtt(
    float *x, float *q, float *att, float *keyCache, float *valueCache,
    const unsigned pos, const unsigned int nHeads, const unsigned int nHeads0, const unsigned int nKvHeads, const unsigned int kvDim0, const unsigned int headSize, const unsigned int seqLen,
    const NnSize nThreads, const NnSize threadIndex) 
{
    SPLIT_THREADS(h0Start, h0End, nHeads0, nThreads, threadIndex);
    unsigned int kvMul = nHeads / nKvHeads;
    float headSizeRoot = sqrtf(headSize);

    for (unsigned int h0 = h0Start; h0 < h0End; h0++) {
        float *hQ = q + h0 * headSize;
        float *hAtt = att + h0 * seqLen;

        for (unsigned int t = 0; t <= pos; t++) {
            float *k = keyCache + t * kvDim0 + (h0 / kvMul) * headSize;
            float score = dotProductF32(hQ, k, headSize) / headSizeRoot;
            hAtt[t] = score;
        }

        softmaxF32(hAtt, pos + 1);

        float *hX = x + h0 * headSize;
        std::memset(hX, 0, headSize * sizeof(float));
        for (unsigned int t = 0; t <= pos; t++) {
            float *hV = valueCache + t * kvDim0 + (h0 / kvMul) * headSize;
            float a = hAtt[t];
            for (int i = 0; i < headSize; i++) {
                hX[i] += a * hV[i];
            }
        }
    }
}

static void mul_F32(float *output, const float *x, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
    assert(n % 4 == 0);
    NnSize nBlocks = n / 4;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

#if defined(__ARM_NEON)
    for (NnSize i = start; i < end; i++) {
        const float *xi = &x[i * 4];
        float *oi = &output[i * 4];
        float32x4_t x_vec = vld1q_f32(xi);
        float32x4_t o_vec = vld1q_f32(oi);
        o_vec = vmulq_f32(o_vec, x_vec);
        vst1q_f32(oi, o_vec);
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

static void copy(NnByte *output, const NnByte *x, NnSize size, const NnSize nThreads, const NnSize threadIndex) {
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
        copy(
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

static void matmulForward_F32_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->weightSize.y);
    ASSERT_EQ(context->outputSize.x, context->weightSize.x);

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

static void matmulForward_F32_Q40_Q80(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    const NnBlockQ40 *weight = (NnBlockQ40 *)context->weight;
    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        NnBlockQ80 *output = (NnBlockQ80 *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        matmul_F32_Q40_Q80(
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
        siluF32(output, context->outputSize.x, nThreads, threadIndex);
    }
}

static void geluForward_F32_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->weightSize.nBytes, 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        geluF32(output, context->outputSize.x, nThreads, threadIndex);
    }
}

static float ropeLlama31Scale(float freq, const NnRopeLlama31OpConfig *config) {
    float waveLen = 2.0f * M_PI * freq;
    float lowFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingLowFreqFactor;
    float highFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingHighFreqFactory;
    if (waveLen < highFreqWavelen) {
        return freq;
    } else if (waveLen > lowFreqWavelen) {
        return freq / config->ropeScalingFactor;
    } else {
        float smooth = (config->ropeScalingOrigMaxSeqLen / waveLen - config->ropeScalingLowFreqFactor) / (config->ropeScalingHighFreqFactory - config->ropeScalingLowFreqFactor);
        return (1 - smooth) * freq / config->ropeScalingFactor + smooth * freq;
    }
}

static void ropeLlama31Forward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    const NnRopeLlama31OpConfig *config = (NnRopeLlama31OpConfig *)context->opConfig;
    const NnRopeSlice *slice = &config->slice;
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

    const unsigned int dim0Half = (config->isQ ? slice->qDim0 : slice->kvDim0) / 2;
    const unsigned int shift = config->isQ ? slice->qShift : 0;
    SPLIT_THREADS(s, e, dim0Half, nThreads, threadIndex);
    const unsigned int iStart = s * 2;
    const unsigned int iEnd = e * 2;

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *x = (float *)context->input[batchIndex];
        NnSize pos = (NnSize)positions[batchIndex];

        for (unsigned int i = iStart; i < iEnd; i += 2) {
            const unsigned int headDim = i % slice->headSize;
            const float freq = 1.0f / powf(slice->ropeTheta, headDim / (float)slice->headSize);
            const float val = pos * freq;
            const float fcr = cosf(val);
            const float fci = sinf(val);
    
            float v0 = x[i];
            float v1 = x[i + 1];
    
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

        multiheadAtt(i, q, att, keyCache, valueCache, pos,
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

static void castForward_F32_F32(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);

    for (NnSize batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        copy(
            context->output[batchIndex],
            context->input[batchIndex],
            context->outputSize.x * sizeof(float),
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

NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType) {
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
        if (quantType == F32_Q40_Q80) return matmulForward_F32_Q40_Q80;
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
        if (quantType == F32_F32_F32) return castForward_F32_F32;
        if (quantType == F32_F32_Q80) return castForward_F32_Q80;
        if (quantType == Q80_Q80_F32) return castForward_Q80_F32;
    }
    return nullptr;
}
