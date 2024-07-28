#include <cmath>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include "common/pthread.h"
#include "funcs.hpp"
#include "utils.hpp"

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__AVX2__)
    #include <immintrin.h>
#endif

#if defined(__AVX2__)
    #define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

    static inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
        // Load 16 bytes from memory
        __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
        __m128i tmph = _mm_srli_epi16(tmpl, 4);
        const __m128i lowMask = _mm_set1_epi8(0xF);
        tmpl = _mm_and_si128(lowMask, tmpl);
        tmph = _mm_and_si128(lowMask, tmph);
        return MM256_SET_M128I(tmph, tmpl);
    }

    static inline float hsum_float_8(const __m256 x) {
        __m128 res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        return _mm_cvtss_f32(res);
    }

    // add int16_t pairwise and return as float vector
    static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
        const __m128i ones = _mm_set1_epi16(1);
        const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
        const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
        const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
        return _mm256_cvtepi32_ps(summed_pairs);
    }

    // multiply int8_t, add results pairwise twice and return as float vector
    static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
        const __m128i xl = _mm256_castsi256_si128(x);
        const __m128i xh = _mm256_extractf128_si256(x, 1);
        const __m128i yl = _mm256_castsi256_si128(y);
        const __m128i yh = _mm256_extractf128_si256(y, 1);
        // Get absolute values of x vectors
        const __m128i axl = _mm_sign_epi8(xl, xl);
        const __m128i axh = _mm_sign_epi8(xh, xh);
        // Sign the values of the y vectors
        const __m128i syl = _mm_sign_epi8(yl, xl);
        const __m128i syh = _mm_sign_epi8(yh, xh);
        // Perform multiplication and create 16-bit values
        const __m128i dotl = _mm_maddubs_epi16(axl, syl);
        const __m128i doth = _mm_maddubs_epi16(axh, syh);
        return sum_i16_pairs_float(doth, dotl);
    }
#endif

void softmax(float* x, const unsigned int size) {
    float maxVal;
#if defined(__ARM_NEON)
    float32x4_t fs;
    float32x4_t fmaxv = vld1q_f32(&x[0]);
    for (unsigned int i = 4; i < size; i += 4) {
        fs = vld1q_f32(&x[i]);
        fmaxv = vmaxq_f32(fmaxv, fs);
    }
    maxVal = vmaxvq_f32(fmaxv);
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
    // normalize
    for (unsigned int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float rms(const float* x, const unsigned int size) {
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
    __m256 a, u;
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
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    return ss;
}

void rmsnorm(float* o, const float* x, const float ms, const float* weight, const unsigned int size, const unsigned int nThreads, const unsigned int threadIndex) {
    SPLIT_RANGE_TO_THREADS(start, end, 0, size, nThreads, threadIndex);

#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fw;
    float32x4_t fx;
    float32x4_t fss = vmovq_n_f32(ms);
    for (unsigned int j = start; j < end; j += 4) {
        fw = vld1q_f32(&weight[j]);
        fx = vld1q_f32(&x[j]);
        fx = vmulq_f32(fx, fw);
        fx = vmulq_f32(fx, fss);
        vst1q_f32(&o[j], fx);
    }
#else
    for (unsigned int j = start; j < end; j++) {
        o[j] = weight[j] * (ms * x[j]);
    }
#endif
}

struct MatmulThreadInfo {
    dl_thread handler;
    float* output;
    const void* input;
    const void* weights;
    unsigned int n;
    unsigned int ds;
    unsigned int de;
};

void matmulF32(const MatmulThreadInfo* a) {
    const float* input = (float*)a->input;
    float* w = (float*)a->weights;
    unsigned int d, j;

#if defined(__ARM_NEON)
    assert(a->n % 4 == 0);
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (d = a->ds; d < a->de; d++) {
        z = vmovq_n_f32(0);
        for (j = 0; j < a->n; j += 4) {
            q = vld1q_f32(&input[j]);
            p = vld1q_f32(&w[d * a->n + j]);
            z = vfmaq_f32(z, q, p);
        }
        a->output[d] = vaddvq_f32(z);
    }
#elif defined(__AVX2__)
    assert(a->n % 8 == 0);
    __m256 a0, b0, u;
    for (d = a->ds; d < a->de; d++) {
        u = _mm256_set1_ps(0.0f);
        for (j = 0; j < a->n; j += 8) {
            a0 = _mm256_loadu_ps(&input[j]);
            b0 = _mm256_loadu_ps(&w[d * a->n + j]);
            u = _mm256_fmadd_ps(a0, b0, u);
        }
        a->output[d] = hsum_float_8(u);
    }
#else
    for (d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (j = 0; j < a->n; j++) {
            val += w[d * a->n + j] * input[j];
        }
        a->output[d] = val;
    }
#endif
}

void matmulF16(const MatmulThreadInfo* a) {
    const float* input = (float*)a->input;
    const uint16_t* w = (uint16_t*)a->weights;
    for (unsigned int d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (unsigned int j = 0; j < a->n; j++) {
            float ww = convertF16ToF32(w[d * a->n + j]);
            val += ww * input[j];
        }
        a->output[d] = val;
    }
}

void matmulQ40(const MatmulThreadInfo* a) {
    const int blocksPerRow = 8;
    const int k = QK40 * blocksPerRow;
    BlockQ40* w = (BlockQ40*)a->weights;
    assert(a->n % k == 0);
    const float* input = (float*)a->input;
    const int n = a->n / k;
    float group[k];

#if defined(__ARM_NEON)
    assert(k % 16 == 0);
    float32x4_t a0;
    float32x4_t b0;
    float32x4_t u;
    for (unsigned int d = a->ds; d < a->de; d++) {
        u = vmovq_n_f32(0);
        for (unsigned int j = 0; j < n; j++) {
            dequantizeQ40Row(&w[d * n * blocksPerRow + j * blocksPerRow], group, k);
            for (unsigned int z = 0; z < k; z += 4) {
                a0 = vld1q_f32(&input[j * k + z]);
                b0 = vld1q_f32(&group[z]);
                u = vfmaq_f32(u, a0, b0);
            }
        }
        a->output[d] = vaddvq_f32(u);
    }
#elif defined(__AVX2__)
    assert(k % 32 == 0);
    __m256 a0, b0, u;
    for (unsigned int d = a->ds; d < a->de; d++) {
        u = _mm256_set1_ps(0.0f);
        for (unsigned int j = 0; j < n; j++) {
            dequantizeQ40Row(&w[d * n * blocksPerRow + j * blocksPerRow], group, k);
            for (unsigned int z = 0; z < k; z += 8) {
                a0 = _mm256_loadu_ps(&input[j * k + z]);
                b0 = _mm256_loadu_ps(&group[z]);
                u = _mm256_fmadd_ps(a0, b0, u);
            }
        }
        a->output[d] = hsum_float_8(u);
    }
#else
    for (unsigned int d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (unsigned int j = 0; j < n; j++) {
            dequantizeQ40Row(&w[d * n * blocksPerRow + j * blocksPerRow], group, k);
            for (unsigned int z = 0; z < k; z++) {
                val += group[z] * input[j * k + z];
            }
        }
        a->output[d] = val;
    }
#endif
}

void matmulQ80(const MatmulThreadInfo* a) {
    float* input = (float*)a->input;
    const BlockQ80* weights = (BlockQ80*)a->weights;
    assert(a->n % QK80 == 0);
    const unsigned int nb = a->n / QK80;

    for (unsigned int d = a->ds; d < a->de; d++) {
        float sum = 0.0;
        for (unsigned int i = 0; i < nb; i++) {
            float s = 0.0;
            for (unsigned int j = 0; j < QK80; j++) {
                s += input[i * QK80 + j] * (float)weights[d * nb + i].qs[j];
            }
            sum += s * convertF16ToF32(weights[d * nb + i].d);
        }
        a->output[d] = sum;
    }
}

void matmulQ40vQ80(const MatmulThreadInfo* a) {
    const BlockQ40* w = (BlockQ40*)a->weights;
    const BlockQ80* input = (BlockQ80*)a->input;
    assert(a->n % QK40 == 0);
    const unsigned int n = a->n / QK40;

#if defined(__ARM_NEON)
    float32x4_t sumv0;
    float32x4_t sumv1;
    for (unsigned int d = a->ds; d < a->de; d++) {
        sumv0 = vmovq_n_f32(0);
        sumv1 = vmovq_n_f32(0);
        for (unsigned int j = 0; j < n; j += 2) {
            const BlockQ40* x0 = &w[d * n + j];
            const BlockQ40* x1 = &w[d * n + j + 1];
            const BlockQ80* y0 = &input[j];
            const BlockQ80* y1 = &input[j + 1];

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

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), convertF16ToF32(x0->d)*convertF16ToF32(y0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), convertF16ToF32(x1->d)*convertF16ToF32(y1->d));
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

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), convertF16ToF32(x0->d) * convertF16ToF32(y0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), convertF16ToF32(x1->d) * convertF16ToF32(y1->d));
#endif
        }
        a->output[d] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
    }
#elif defined(__AVX2__)
    for (unsigned int d = a->ds; d < a->de; d++) {
        __m256 acc = _mm256_setzero_ps();

        for (unsigned int j = 0; j < n; j++) {
            /* Compute combined scale for the block */
            const __m256 cd = _mm256_set1_ps( convertF16ToF32(w[d * n + j].d) * convertF16ToF32(input[j].d) );

            __m256i bx = bytes_from_nibbles_32(w[d * n + j].qs);

            // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
            const __m256i off = _mm256_set1_epi8( 8 );
            bx = _mm256_sub_epi8(bx, off);

            __m256i by = _mm256_loadu_si256((const __m256i *)input[j].qs);

            const __m256 q = mul_sum_i8_pairs_float(bx, by);

            /* Multiply q with scale and accumulate */
            acc = _mm256_fmadd_ps( cd, q, acc );
        }

        a->output[d] = hsum_float_8(acc);
    }
#else
    float group[QK40];
    for (unsigned int d = a->ds; d < a->de; d++) {
        float sum = 0.0;
        for (unsigned int j = 0; j < n; j++) {
            dequantizeQ40Row(&w[d * n + j], group, QK40);
            float iD = convertF16ToF32(input[j].d);
            for (unsigned int z = 0; z < QK40; z++) {
                sum += group[z] * iD * (float)input[j].qs[z];
            }
        }
        a->output[d] = sum;
    }
#endif
}

void matmulQ80vQ80(const MatmulThreadInfo* a) {
    const BlockQ80* input = (BlockQ80*)a->input;
    BlockQ80* weights = (BlockQ80*)a->weights;
    assert(a->n % QK80 == 0);
    const unsigned int nb = a->n / QK80;

    for (unsigned int d = a->ds; d < a->de; d++) {
        float sum = 0.0;
        for (unsigned int i = 0; i < nb; i++) {
            int s = 0;
            for (unsigned int j = 0; j < QK80; j++) {
                s += input[i].qs[j] * (int)weights[d * nb + i].qs[j];
            }
            sum += s * (convertF16ToF32(input[i].d) * convertF16ToF32(weights[d * nb + i].d));
        }
        a->output[d] = sum;
    }
}

//     weights      input    output
//   ___________     ___      ___
//   |         |     | |      | |
// d |         | *   | |  = d | |
//   |_________|   n | |      |_|
//        n          |_|       1
//                    1
void matmul(const FloatType weightsFloatType, const FloatType inputFloatType, float* output, const void* input, const void* weights, const unsigned int n, const unsigned int d, const unsigned int nThreads, const unsigned int threadIndex) {
    SPLIT_RANGE_TO_THREADS(ds, de, 0, d, nThreads, threadIndex);

    MatmulThreadInfo s;
    s.output = output;
    s.input = input;
    s.weights = weights;
    s.n = n;
    s.ds = ds;
    s.de = de;

    if (inputFloatType == F32) {
        if (weightsFloatType == F32) {
            matmulF32(&s);
            return;
        }
        if (weightsFloatType == F16) {
            matmulF16(&s);
            return;
        }
        if (weightsFloatType == Q40) {
            matmulQ40(&s);
            return;
        }
        if (weightsFloatType == Q80) {
            matmulQ80(&s);
            return;
        }
    } else if (inputFloatType == Q80) {
        if (weightsFloatType == Q40) {
            matmulQ40vQ80(&s);
            return;
        }
        if (weightsFloatType == Q80) {
            matmulQ80vQ80(&s);
            return;
        }
    }

    printf("Unsupported float types: %d/%d\n", weightsFloatType, inputFloatType);
    exit(EXIT_FAILURE);
}

float dotProduct(const float* a, const float* b, const unsigned int size) {
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

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

void gelu(float* t, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex) {
    SPLIT_RANGE_TO_THREADS(start, end, 0, n, nThreads, threadIndex);

    for (unsigned int i = start; i < end; i++) {
        float x = t[i];
        t[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

void silu(float* t, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex) {
    SPLIT_RANGE_TO_THREADS(start, end, 0, n, nThreads, threadIndex);

    for (unsigned int i = start; i < end; i++) {
        float x = t[i];
        t[i] = x / (1.0f + expf(-x));
    }
}

void mul(float* output, const float* input, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex) {
    SPLIT_RANGE_TO_THREADS(start, end, 0, n, nThreads, threadIndex);

    for (unsigned int i = start; i < end; i++) {
        output[i] *= input[i];
    }
}

void mulScalar(float* output, const float c, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex) {
    SPLIT_RANGE_TO_THREADS(start, end, 0, n, nThreads, threadIndex);

    for (unsigned int i = start; i < end; i++) {
        output[i] *= c;
    }
}

void add(float* output, const float* input, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex) {
    SPLIT_RANGE_TO_THREADS(start, end, 0, n, nThreads, threadIndex);

    for (unsigned int i = start; i < end; i++) {
        output[i] += input[i];
    }
}
