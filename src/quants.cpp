#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cassert>
#include "quants.hpp"

#define NEON 1

#if NEON
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
            assert(n % QK40 == 0);
            int blocks = n / QK40 * d;
            return blocks * sizeof(BlockQ40);
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

void dequantizeQ40Row(const BlockQ40* x, float* y, int k) {
    static const int qk = QK40;
    assert(k % qk == 0);
    const int nb = k / qk;

#if NEON
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

void initQuants() {
    initF16ToF32();
}
