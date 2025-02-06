#ifndef NN_FP16_H
#define NN_FP16_H

#include <cassert>
#include <cstring>
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

typedef uint16_t NnFp16;

static float convertF16toF32(const NnFp16 value) {
#if defined(__ARM_NEON)
    __fp16 fp;
    std::memcpy(&fp, &value, sizeof(fp));
    return (float)fp;
#else
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
#endif
}
    
static NnFp16 convertF32ToF16(const float x) {
#if defined(__ARM_NEON)
    __fp16 h = x;
    return *(NnFp16 *)&h;
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

#endif