#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__AVX2__) || defined(__AVX512F__)
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

#if defined(__ARM_NEON)
static inline float32x4_t expf_neon(float32x4_t x) {
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

#if defined(__AVX2__)
static inline float horizontalSum_avx2(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float horizontalMax_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(v_low, v_high);
    __m128 max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    __m128 max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(max32);
}

static inline __m256 expf_avx2(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    __m256 y = _mm256_mul_ps(x, log2e);
    __m256i n = _mm256_cvtps_epi32(y);
    __m256 n_float = _mm256_cvtepi32_ps(n);
    __m256 f = _mm256_sub_ps(y, n_float);
    __m256 p = c4;
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    __m256i exponent = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    exponent = _mm256_slli_epi32(exponent, 23);
    __m256 two_n = _mm256_castsi256_ps(exponent);
    return _mm256_mul_ps(p, two_n);
}
#endif

static float invRms_F32(const float *x, const unsigned int size, const float epsilon) {
    float sum;
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fsq;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int j = 0; j < size; j += 4) {
        fsq = vld1q_f32(&x[j]);
        fs = vmlaq_f32(fs, fsq, fsq);
    }
    sum = vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int j = 0; j < size; j += 8) {
        a = _mm256_loadu_ps(&x[j]);
        u = _mm256_fmadd_ps(a, a, u);
    }
    sum = horizontalSum_avx2(u);
#else
    sum = 0;
    for (unsigned int j = 0; j < size; j++) {
        sum += x[j] * x[j];
    }
#endif
    sum /= size;
    sum += epsilon;
    return 1.0f / sqrtf(sum);
}

static void rmsNorm_F32(float *output, const float *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);
    float32x4_t fw;
    float32x4_t fx;
    float32x4_t fss = vmovq_n_f32(invRms);
    for (; i < neonEnd; i += 4) {
        fw = vld1q_f32(&w[i]);
        fx = vld1q_f32(&x[i]);
        fx = vmulq_f32(fx, fw);
        fx = vmulq_f32(fx, fss);
        vst1q_f32(&output[i], fx);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    const __m256 invRmsVec = _mm256_set1_ps(invRms);
    for (; i < avxEnd; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]);
        __m256 wVec = _mm256_loadu_ps(&w[i]);
        __m256 scaledX = _mm256_mul_ps(xVec, invRmsVec);
        __m256 result = _mm256_mul_ps(scaledX, wVec);
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++)
        output[i] = w[i] * (invRms * x[i]);
}

static void rmsNorm_Q80_F32_F32(float *output, const NnBlockQ80 *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    assert(size % Q80_BLOCK_SIZE == 0);
    const NnUint nBlocks = size / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

    for (NnUint i = start; i < end; i++) {
        float d = CONVERT_F16_TO_F32(x[i].d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            output[k] = w[k] * (invRms * d * x[i].qs[j]);
        }
    }
}

static void matmul_F32_F32_F32(float *output, const float *x, const float *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
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
        output[i] = horizontalSum_avx2(u);
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

static void matmul_Q80_Q40_F32(float *output, const NnBlockQ80 *x, const NnBlockQ40 *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
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

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), CONVERT_F16_TO_F32(w2->d) * CONVERT_F16_TO_F32(x2->d));
            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), CONVERT_F16_TO_F32(w3->d) * CONVERT_F16_TO_F32(x3->d));
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

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
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
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), s);
        }

        output[di] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    }
#elif defined(__AVX512F__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w8 = _mm_loadu_si128((const __m128i*)wb->qs);
            __m128i v_w0 = _mm_and_si128(w8, _mm_set1_epi8(0x0F));
            __m128i v_w1 = _mm_srli_epi16(w8, 4);
            v_w1 = _mm_and_si128(v_w1, _mm_set1_epi8(0x0F));
            
            v_w0 = _mm_sub_epi8(v_w0, _mm_set1_epi8(8));
            v_w1 = _mm_sub_epi8(v_w1, _mm_set1_epi8(8));

            __m256i w8_combined = _mm256_set_m128i(v_w1, v_w0);
            __m512i w16 = _mm512_cvtepi8_epi16(w8_combined);

            __m256i x8 = _mm256_loadu_si256((const __m256i*)xb->qs);
            __m512i x16 = _mm512_cvtepi8_epi16(x8);

            __m512i products = _mm512_madd_epi16(w16, x16);
            sum += _mm512_reduce_add_epi32(products) * s;
        }
        output[i] = sum;
    }
#elif defined(__AVX2__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w_packed = _mm_loadu_si128((const __m128i*)wb->qs);

            __m128i w0_low = _mm_and_si128(w_packed, _mm_set1_epi8(0x0F));
            __m128i w0 = _mm_sub_epi8(w0_low, _mm_set1_epi8(8));

            __m128i w1_high = _mm_srli_epi16(w_packed, 4);
            w1_high = _mm_and_si128(w1_high, _mm_set1_epi8(0x0F));
            __m128i w1 = _mm_sub_epi8(w1_high, _mm_set1_epi8(8));

            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);

            __m128i i1_8 = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i i2_8 = _mm_loadu_si128((const __m128i*)(xb->qs + Q40_BLOCK_SIZE / 2));

            __m256i i1_16 = _mm256_cvtepi8_epi16(i1_8);
            __m256i i2_16 = _mm256_cvtepi8_epi16(i2_8);

            __m256i prod0 = _mm256_mullo_epi16(w0_16, i1_16);
            __m256i prod1 = _mm256_mullo_epi16(w1_16, i2_16);
            __m256i sum_prod = _mm256_add_epi16(prod0, prod1);

            __m256i ones = _mm256_set1_epi16(1);
            __m256i sum32 = _mm256_madd_epi16(sum_prod, ones);

            __m128i sum_low = _mm256_castsi256_si128(sum32);
            __m128i sum_high = _mm256_extracti128_si256(sum32, 1);
            sum_low = _mm_add_epi32(sum_low, sum_high);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            int32_t block_sum = _mm_extract_epi32(sum_low, 0);

            sum += block_sum * s;
        }
        output[i] = sum;
    }
#else
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                const int w0 = (wb->qs[k] & 0x0F) - 8;
                const int w1 = (wb->qs[k] >> 4) - 8;
                const int i1 = xb->qs[k];
                const int i2 = xb->qs[k + Q80_BLOCK_SIZE / 2];
                sum += (w0 * i1 + w1 * i2) * s;
            }
        }
        output[i] = sum;
    }
#endif
}

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

static void gelu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

static void silu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);

    for (; i < neonEnd; i += 4) {
        float32x4_t x = vld1q_f32(&output[i]);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t exp_negx = expf_neon(neg_x);
        float32x4_t denominator = vaddq_f32(exp_negx, vdupq_n_f32(1.0f));

        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(recip, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(denominator, recip)));

        float32x4_t result = vmulq_f32(x, recip);
        vst1q_f32(output + i, result);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);

    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (; i < avxEnd; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(output + i);
        __m256 neg_x = _mm256_sub_ps(zero, x_vec);
        __m256 exp_negx = expf_avx2(neg_x);
        __m256 denominator = _mm256_add_ps(ones, exp_negx);
        __m256 result = _mm256_div_ps(x_vec, denominator);
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++) {
        float x = output[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

static void add_F32(float *output, const float *x, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        output[i] += x[i];
    }
}

static void add_Q80_F32(float *y, const NnBlockQ80 *x, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

#if defined(__ARM_NEON)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
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
#elif defined(__AVX2__)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 8) {
            __m128i i8_vec = _mm_loadl_epi64((const __m128i*)(xi->qs + j));

            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);

            __m256 f_vec = _mm256_cvtepi32_ps(i32_vec);

            __m256 scale = _mm256_set1_ps(xid);
            __m256 scaled = _mm256_mul_ps(f_vec, scale);

            float* y_ptr = y + i * Q80_BLOCK_SIZE + j;
            __m256 y_vec = _mm256_loadu_ps(y_ptr);
            y_vec = _mm256_add_ps(y_vec, scaled);
            _mm256_storeu_ps(y_ptr, y_vec);
        }
    }
#else
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j++) {
            y[i * Q80_BLOCK_SIZE + j] += xid * xi->qs[j];
        }
    }
#endif
}

void softmax_F32(float *x, const NnUint size) {
    if (size == 0)
        return;

#if defined(__ARM_NEON)
    NnUint j;
    float maxVal;
    if (size >= 4) {
        float32x4_t fs;
        float32x4_t fmaxv = vld1q_f32(&x[0]);
        j = size - (size % 4);
        for (NnUint i = 4; i < j; i += 4) {
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
    NnUint i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vsubq_f32(val, maxVal_vec);
        val = expf_neon(val);
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
#elif defined(__AVX2__)
    float maxVal;
    const unsigned avxEnd = size - (size % 8);
    NnUint i = 0;

    if (avxEnd >= 8) {
        __m256 max_vec = _mm256_loadu_ps(x);
        for (; i < avxEnd; i += 8) {
            __m256 vec = _mm256_loadu_ps(&x[i]);
            max_vec = _mm256_max_ps(max_vec, vec);
        }
        maxVal = horizontalMax_avx2(max_vec);

        for (; i < size; ++i) {
            if (x[i] > maxVal)
                maxVal = x[i];
        }
    } else {
        maxVal = x[0];
        i = 1;
    }
    for (; i < size; ++i) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }

    __m256 max_val_vec = _mm256_set1_ps(maxVal);
    __m256 sum_vec = _mm256_setzero_ps();
    float sum = 0.0f;
    
    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(&x[i]);
        vec = _mm256_sub_ps(vec, max_val_vec);
        vec = expf_avx2(vec);
        _mm256_storeu_ps(&x[i], vec);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    sum = horizontalSum_avx2(sum_vec);
    for (; i < size; ++i) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0)
        sum = 0.000001;

    const float inv_sum = 1.0f / sum;
    const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(x + i);
        vec = _mm256_mul_ps(vec, inv_sum_vec);
        _mm256_storeu_ps(x + i, vec);
    }
    for (; i < size; i++)
        x[i] *= inv_sum;
#else
    float maxVal = x[0];
    for (NnUint i = 1; i < size; i++) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }
    float sum = 0.0f;
    for (NnUint i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    if (sum == 0.0)
        sum = 0.000001;
    for (NnUint i = 0; i < size; i++)
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
    return horizontalSum_avx2(u);
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
    const NnUint pos, const NnUint nHeads, const NnUint nHeads0, const NnUint nKvHeads, const NnUint kvDim0, const NnUint headSize, const NnUint seqLen,
    const NnUint nThreads, const NnUint threadIndex) 
{
    SPLIT_THREADS(h0Start, h0End, nHeads0, nThreads, threadIndex);
    const NnUint kvMul = nHeads / nKvHeads;
    const float headSizeRoot = sqrtf(headSize);

    for (NnUint h0 = h0Start; h0 < h0End; h0++) {
        const float *hQ = &q[h0 * headSize];
        const NnUint headIndex = h0 / kvMul;
        const float *hKc = &keyCache[headIndex * headSize];
        const float *hVc = &valueCache[headIndex * headSize];
        float *hAtt = &att[h0 * seqLen];

        for (NnUint t = 0; t <= pos; t++) {
            const float *posK = &hKc[t * kvDim0];
            const float score = dotProduct_F32(hQ, posK, headSize) / headSizeRoot;
            hAtt[t] = score;
        }

        softmax_F32(hAtt, pos + 1);

        float *hX = &x[h0 * headSize];
        std::memset(hX, 0, headSize * sizeof(float));

        for (NnUint t = 0; t <= pos; t++) {
            const float *posV = &hVc[t * kvDim0];
            const float posA = hAtt[t];
            for (int i = 0; i < headSize; i++) {
                hX[i] += posA * posV[i];
            }
        }
    }
}

static void mul_F32(float *y, const float *x, const float *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;

#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 8);
    for (; i < neonEnd; i += 4) {
        float32x4_t out_vec = vld1q_f32(&x[i]);
        float32x4_t x_vec = vld1q_f32(&m[i]);
        float32x4_t res_vec = vmulq_f32(out_vec, x_vec);
        vst1q_f32(&y[i], res_vec);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    for (; i < avxEnd; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&x[i]);
        __m256 x_vec = _mm256_loadu_ps(&m[i]);
        __m256 res_vec = _mm256_mul_ps(out_vec, x_vec);
        _mm256_storeu_ps(&y[i], res_vec);
    }
#endif
    for (; i < end; i++)
        y[i] = x[i] * m[i];
}

static void mul_Q80_F32(float *y, const float *x, const NnBlockQ80 *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++) {
        const NnBlockQ80 *b = &m[i];
        float d = CONVERT_F16_TO_F32(b->d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            y[k] = x[k] * d * b->qs[j];
        }
    }
}

static void copy_UNK(NnByte *output, const NnByte *x, NnSize size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    NnUint s = end - start;
    if (s != 0)
        std::memcpy(&output[start], &x[start], s);
}

//

static void mergeAddForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnUint nSlices = context->inputSize.x / context->outputSize.x;

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        float *input = (float *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
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

static void mergeAddForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->inputSize.floatType == F_Q80);
    assert(context->outputSize.floatType == F_32);

    NnUint nSlices = context->inputSize.x / context->outputSize.x;
    NnUint xSize = context->outputSize.x / Q80_BLOCK_SIZE;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            add_Q80_F32(
                output,
                &input[sliceIndex * xSize],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void initEmbeddingForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, 1);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.x, context->outputSize.x);
}

static void embeddingForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnUint token = (NnUint)*((float *)context->input[batchIndex]);
        copy_UNK(
            context->output[batchIndex],
            &context->weight[token * dimSize],
            dimSize,
            nThreads,
            threadIndex);
    }
}

static void embeddingForward_F32_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnUint token = (NnUint)*((float *)context->input[batchIndex]);
        quantizeF32toQ80(
            (float *)&context->weight[token * dimSize],
            (NnBlockQ80 *)context->output[batchIndex],
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void invRmsForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (threadIndex == 0) {
        ASSERT_EQ(context->inputSize.y, context->nBatches);
        ASSERT_EQ(context->outputSize.x, 1);
        ASSERT_EQ(context->outputSize.y, context->nBatches);

        const NnInvRmsOpConfig *config = (NnInvRmsOpConfig *)context->opConfig;
        for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            float *input = (float *)context->input[batchIndex];
            float *output = (float *)context->output[batchIndex];
            DEBUG_VECTOR(context, "input", input);
            float rms = invRms_F32(
                input,
                context->inputSize.x,
                config->epsilon);
            output[0] = rms;
            DEBUG_SCALAR(context, "output", rms);
        }
    }
}

static void initRmsNormForward_ANY_F32_F32(NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    NnBufferConfig *rmsBufferConfig = &context->bufferConfigs[config->invRmsBufferIndex];
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.floatType, F_32);
    ASSERT_EQ(context->weightSize.y, 1);
    ASSERT_EQ(context->weightSize.x, context->inputSize.x);
    ASSERT_EQ(rmsBufferConfig->size.floatType, F_32);
    ASSERT_EQ(rmsBufferConfig->size.x, 1);
    ASSERT_EQ(rmsBufferConfig->size.y, context->nBatches);
}

static void rmsNormForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);

    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    const float *weight = (float *)context->weight;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_F32(
            output,
            input,
            invRms[batchIndex],
            weight,
            context->inputSize.x,
            nThreads,
            threadIndex);
    }
}

static void rmsNormForward_Q80_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);

    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    const float *weight = (float *)context->weight;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_Q80_F32_F32(
            output,
            input,
            invRms[batchIndex],
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

static bool matmulForward_llamafile(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (batchSize == 1 || !context->hasInputContinuousMemory || !context->hasOutputContinuousMemory)
        return false;

    const NnUint n = context->weightSize.y / getBlockSize(context->inputSize.floatType);
    const NnUint d = context->weightSize.x;
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

static void matmulForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const float *weight = (float *)context->weight;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
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
        DEBUG_VECTOR(context, "output", output);
    }
}

static void matmulForward_Q80_Q40_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const NnBlockQ40 *weight = (NnBlockQ40 *)context->weight;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
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

static void siluForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        silu_F32(output, context->outputSize.x, nThreads, threadIndex);
    }
}

static void geluForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        gelu_F32(output, context->outputSize.x, nThreads, threadIndex);
    }
}

static void initRopeLlama3Forward(NnCpuOpContext *context) {
    const NnRopeLlamaOpConfig *config = (NnRopeLlamaOpConfig *)context->opConfig;
    if (context->bufferFlags[config->ropeCacheBufferIndex] == 1)
        return;
    context->bufferFlags[config->ropeCacheBufferIndex] = 1;

    float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    fullfillRopeLlama3Cache(config, cache);
}

static void ropeLlamaForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnRopeLlamaOpConfig *config = (NnRopeLlamaOpConfig *)context->opConfig;
    const NnRopeSlice *slice = &config->slice;

    const NnUint dim0Half = (config->isQ ? slice->qDim0 : slice->kvDim0) / 2;
    const NnUint shift = config->isQ ? slice->qShift : 0;
    SPLIT_THREADS(s, e, dim0Half, nThreads, threadIndex);
    const NnUint iStart = s * 2;
    const NnUint iEnd = e * 2;

    const float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *x = (float *)context->input[batchIndex];
        const NnUint pos = (NnUint)positions[batchIndex];
        const float *posCache = &cache[pos * slice->sliceDim + shift];

        for (NnUint i = iStart; i < iEnd; i += 2) {
            const float fcr = posCache[i];
            const float fci = posCache[i + 1];
            const float v0 = x[i];
            const float v1 = x[i + 1];

            float x0 = v0 * fcr - v1 * fci;
            float x1 = v0 * fci + v1 * fcr;
            x[i] = x0;
            x[i + 1] = x1;
        }
    }
}

static void initMultiHeadAttForward(NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, config->qSliceD0);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    NnSize2D *querySize = &context->bufferConfigs[config->queryBufferIndex].size;
    ASSERT_EQ(querySize->x, config->qSliceD0);
    NnSize2D *posSize = &context->pipeConfigs[config->positionPipeIndex].size;
    ASSERT_EQ(posSize->x, 1);
    ASSERT_EQ(posSize->y, context->nBatches);
}

static void multiHeadAttForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    float *query = (float *)context->buffers[config->queryBufferIndex];
    float *keyCache = (float *)context->buffers[config->keyCacheBufferIndex];
    float *valueCache = (float *)context->buffers[config->valueCacheBufferIndex];
    float *att = (float *)context->buffers[config->attBufferIndex];
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *i = (float *)context->input[batchIndex];
        float *q = &query[batchIndex * config->qSliceD0];
        NnUint pos = (NnUint)positions[batchIndex];
        assert(pos < config->seqLen);

        DEBUG_VECTOR(context, "input", i);
        DEBUG_VECTOR(context, "q", q);

        multiheadAtt_F32(i, q, 
            &att[batchIndex * config->nHeads0 * config->seqLen],
            keyCache, valueCache, pos,
            config->nHeads, config->nHeads0,
            config->nKvHeads, config->kvDim0, config->headSize, config->seqLen, nThreads, threadIndex);

        DEBUG_VECTOR(context, "output", i);
    }
}

static void mulForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    const float *multiplier = (float *)context->buffers[config->multiplierBufferIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        const float *m = &multiplier[context->inputSize.x * batchIndex];
        mul_F32(
            output,
            input,
            m,
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void mulForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    const NnBlockQ80 *multiplier = (NnBlockQ80 *)context->buffers[config->multiplierBufferIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        const NnBlockQ80 *m = &multiplier[batchIndex * context->inputSize.x / Q80_BLOCK_SIZE];
        mul_Q80_F32(
            output,
            input,
            m,
            context->outputSize.x,
            nThreads,
            threadIndex);
    }
}

static void initCastForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
}

static void castForward_ANY(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnUint rowBytes = context->outputSize.nBytes / context->outputSize.y;

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        copy_UNK(
            context->output[batchIndex],
            context->input[batchIndex],
            rowBytes,
            nThreads,
            threadIndex);
    }
}

static void castForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
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

static void castForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->outputSize.floatType, F_32);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
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

static void shiftForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->hasInputContinuousMemory, true);
    ASSERT_EQ(context->hasOutputContinuousMemory, true);
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, 1);

    const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)context->opConfig;
    const float *indexes = (float *)context->pipes[config->indexPipeIndex];
    const NnSize dimBytes = getBytes(F_32, context->inputSize.x);
    NnByte *output = context->output[0];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const NnSize index = (NnSize)indexes[batchIndex];
        assert((index + 1) * context->inputSize.x <= context->outputSize.x);
        copy_UNK(
            &output[index * dimBytes],
            context->input[batchIndex],
            dimBytes,
            nThreads,
            threadIndex);
    }
}

// device

void printCpuInstructionSet() {
    printf("🧠 CPU:");
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
#if defined(__AVX512F__)
    printf(" avx512f");
#endif
    printf("\n");
}

NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_EMBEDDING)
        return initEmbeddingForward;
    if (code == OP_RMS_NORM)
        return initRmsNormForward_ANY_F32_F32;
    if (code == OP_ROPE_LLAMA)
        return initRopeLlama3Forward;
    if (code == OP_MULTIHEAD_ATT)
        return initMultiHeadAttForward;
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
    if (code == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return invRmsForward_F32_F32;
    }
    if (code == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return rmsNormForward_F32_F32_F32;
        if (quantType == Q80_F32_F32) return rmsNormForward_Q80_F32_F32;
    }
    if (code == OP_MATMUL) {
        if (quantType == F32_F32_F32) return matmulForward_F32_F32_F32;
        if (quantType == Q80_Q40_F32) return matmulForward_Q80_Q40_F32;
    }
    if (code == OP_ROPE_LLAMA) {
        if (quantType == F32_F32_F32) return ropeLlamaForward_F32_F32;
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
        if (quantType == F32_F32_F32) return castForward_ANY;
        if (quantType == F32_F32_Q80) return castForward_F32_Q80;
        if (quantType == Q80_Q80_Q80) return castForward_ANY;
        if (quantType == Q80_Q80_F32) return castForward_Q80_F32;
    }
    if (code == OP_SHIFT) {
        if (quantType == F32_F32_F32) return shiftForward_F32_F32;
    }
    return nullptr;
}
