#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cassert>
#include "quants.hpp"

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

int getNumbersPerBatch(FloatType type) {
    switch (type) {
        case F32:
            return 1;
        case F16:
            return 1;
        case Q40:
            return QK40;
        case Q80:
            return QK80;
        case FUNK:
            break;
    }
    fprintf(stderr, "Unsupported float type %d\n", type);
    exit(EXIT_FAILURE);
}

long getBatchBytes(FloatType type, int n, int d) {
    switch (type) {
        case F32:
            return n * d * sizeof(float);
        case F16:
            return n * d * sizeof(uint16_t);
        case Q40:
            {
                assert(n % QK40 == 0);
                int blocks = n / QK40 * d;
                return blocks * sizeof(BlockQ40);
            }
        case Q80:
            {
                assert(n % QK80 == 0);
                int blocks = n / QK80 * d;
                return blocks * sizeof(BlockQ80);
            }
        case FUNK:
            break;
    }
    fprintf(stderr, "Unsupported float type %d\n", type);
    exit(EXIT_FAILURE);
}

float F16ToF32[65536];

// https://gist.github.com/rygorous/2144712
float _convertF16ToF32(uint16_t value) {
    union F32
    {
        uint32_t u;
        float f;
    };

    const F32 magic = { (254U - 15U) << 23 };
    const F32 was_infnan = { (127U + 16U) << 23 };
    F32 out;

    out.u = (value & 0x7FFFU) << 13;
    out.f *= magic.f;
    if (out.f >= was_infnan.f) {
        out.u |= 255U << 23;
    }
    out.u |= (value & 0x8000U) << 16;
    return out.f;
}

uint16_t _convertF32ToF16(float value) {
    unsigned int fltInt32 = *(unsigned int*)&value;
    unsigned short fltInt16;

    fltInt16 = (fltInt32 >> 31) << 5;
    unsigned short tmp = (fltInt32 >> 23) & 0xff;
    tmp = (tmp - 0x70) & ((unsigned int)((int)(0x70 - tmp) >> 4) >> 27);
    fltInt16 = (fltInt16 | tmp) << 10;
    fltInt16 |= (fltInt32 >> 13) & 0x3ff;
    return fltInt16;
}

void initF16ToF32() {
    for (int i = 0; i < 65536; i++) {
        F16ToF32[i] = _convertF16ToF32(i);
    }
}

float convertF16ToF32(uint16_t value) {
    return F16ToF32[value];
}

// https://github.com/mitsuba-renderer/openexr/blob/dbabb6f9500ee628c1faba21bb8add2649cc32a6/IlmBase/Half/half.cpp#L85
uint16_t convertF32ToF16(const float x) {
    int i = *(int*)&x;
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
    } else if (e == 0xff - (127 - 15)) {
        if (m == 0) {
            return s | 0x7c00;
        } else {
            m >>= 13;
            return s | 0x7c00 | m | (m == 0);
        }
    } else {
        m = m + 0x00000fff + ((m >> 13) & 1);

        if (m & 0x00800000) {
            m =  0;
            e += 1;
        }
        if (e > 30) {
            // overflow (); // TODO: this should not be commented out
            return s | 0x7c00;
        }
        return s | (e << 10) | (m >> 13);
    }
}

void dequantizeQ40Row(const BlockQ40* x, float* y, int k) {
    static const int qk = QK40;
    assert(k % qk == 0);
    const int nb = k / qk;

#if defined(__ARM_NEON)
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t  s8b = vdupq_n_s8(0x8);

    for (int i = 0; i < nb; i++) {
        const BlockQ40* b = &x[i];
        const float d = convertF16ToF32(b->d);

        const uint8x16_t v0_0 = vld1q_u8(b->qs);

        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));

        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);

        int8x8_t r1 = vget_low_s8(v0_0ls);
        int8x8_t r2 = vget_high_s8(v0_0ls);
        int8x8_t r3 = vget_low_s8(v0_0hs);
        int8x8_t r4 = vget_high_s8(v0_0hs);

        for (int j = 0; j < 8; j++) {
            y[i * qk + j + 0] = r1[j] * d;
            y[i * qk + j + 8] = r2[j] * d;
            y[i * qk + j + 16] = r3[j] * d;
            y[i * qk + j + 24] = r4[j] * d;
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        const BlockQ40* b = &x[i];
        const float d = convertF16ToF32(b->d);

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (b->qs[j] & 0x0F) - 8;
            const int x1 = (b->qs[j] >>   4) - 8;

            y[i * qk + j] = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
#endif
}

void quantizeQ80Row(float* input, BlockQ80* output, int k, unsigned int nThreads, unsigned int threadIndex) {
    assert(k % QK80 == 0);

    const int nBlocks = k / QK80;
    const int blocksPerThread = nBlocks / nThreads;
    const int sk = blocksPerThread * QK80;
    const int currentThreadBlocks = blocksPerThread + (threadIndex == nThreads - 1 ? nBlocks % nThreads : 0);

    const float* x = &input[sk * threadIndex];
    BlockQ80* y = &output[blocksPerThread * threadIndex];

#if defined(__ARM_NEON)
    float dBuf[4];

    for (int i = 0; i < currentThreadBlocks; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j] = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

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

        for (int j = 0; j < QK80; j++) {
            const float v = fabsf(x[i*QK80 + j]);
            amax = amax > v ? amax : v;
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = convertF32ToF16(d);

        for (int j = 0; j < QK80; ++j) {
            const float x0 = x[i*QK80 + j]*id;
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

void dequantizeQ80Row(const BlockQ80* input, float* output, int k, unsigned int nThreads, unsigned int threadIndex) {
    assert(k % QK80 == 0);

    const int nBlocks = k / QK80;
    const int blocksPerThread = nBlocks / nThreads;
    const int sk = blocksPerThread * QK80;
    const int currentThreadBlocks = blocksPerThread + (threadIndex == nThreads - 1 ? nBlocks % nThreads : 0);

    const BlockQ80* x = &input[blocksPerThread * threadIndex];
    float* y = &output[sk * threadIndex];

    for (int i = 0; i < currentThreadBlocks; i++) {
        const float d = convertF16ToF32(x[i].d);

        for (int j = 0; j < QK80; ++j) {
            y[i*QK80 + j] = x[i].qs[j]*d;
        }
    }
}

void initQuants() {
    initF16ToF32();
}
