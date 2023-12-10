#include <stdio.h>
#include <stdlib.h>
#include "quants.hpp"
#include <cassert>
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

long getFloatBytes(FloatType type) {
    switch (type) {
        case F32:
            return sizeof(float);
        case F16:
            return sizeof(uint16_t);
        default:
            printf("Unknown float type %d\n", type);
            exit(1);
    }
}

float F16ToF32[65536];

union Fp32
{
    uint32_t u;
    float f;
};

// https://gist.github.com/rygorous/2144712
float _convertF16ToF32(uint16_t value) {
    const Fp32 magic = { (254U - 15U) << 23 };
    const Fp32 was_infnan = { (127U + 16U) << 23 };
    Fp32 out;

    out.u = (value & 0x7FFFU) << 13;
    out.f *= magic.f;
    if (out.f >= was_infnan.f) {
        out.u |= 255U << 23;
    }
    out.u |= (value & 0x8000U) << 16;
    return out.f;
}

void initF16ToF32() {
    for (int i = 0; i < 65536; i++) {
        F16ToF32[i] = _convertF16ToF32(i);
    }
}

float convertF16ToF32(uint16_t value) {
    return F16ToF32[value];
}

void initQuants() {
    initF16ToF32();
}
