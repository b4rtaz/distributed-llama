#include <cstdio>
#include <cmath>
#include <cassert>
#include "quants.hpp"

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

    for (int i = 0; i < nb; i++) {
        const BlockQ40* b = &x[i];
        const float d = convertF16ToF32(b->d);

        for (int j = 0; j < qk / 2; ++j) {
            const int x0 = (b->qs[j] & 0x0F) - 8;
            const int x1 = (b->qs[j] >>   4) - 8;

            y[i * qk + j * 2 + 0] = x0 * d;
            y[i * qk + j * 2 + 1] = x1 * d;
        }
    }
}

void initQuants() {
    initF16ToF32();
}
