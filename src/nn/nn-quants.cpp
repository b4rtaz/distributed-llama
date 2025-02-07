#include "nn-quants.hpp"
#include <cassert>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <cstdio>

float convertF16toF32Impl(const NnFp16 value) {
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

NnFp16 convertF32ToF16Impl(const float x) {
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
}

void quantizeF32toQ80(const float *input, NnBlockQ80 *output, const NnSize k, const NnSize nThreads, const NnSize threadIndex) {
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

void dequantizeQ80toF32(const NnBlockQ80 *input, float* output, const NnSize k, const NnSize nThreads, const NnSize threadIndex) {
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

void quantizeF32toQ40(const float *x, NnBlockQ40 *output, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
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

void dequantizeQ40toF32(const NnBlockQ40 *x, float *output, const NnSize n, const NnSize nThreads, const NnSize threadIndex) {
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

const char *floatTypeToString(NnFloatType type) {
    if (type == F_UNK) return "F_UNK";
    if (type == F_32) return "F_32";
    if (type == F_16) return "F_16";
    if (type == F_Q40) return "F_Q40";
    if (type == F_Q80) return "F_Q80";
    throw std::invalid_argument("Unknown float type");
}
