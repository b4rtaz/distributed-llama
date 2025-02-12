#ifndef NN_QUANTS_H
#define NN_QUANTS_H

#include <cstdint>
#include <cstring>
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

typedef std::uint8_t NnByte;
typedef std::uint32_t NnSize;
typedef std::uint16_t NnFp16;

float convertF16toF32Impl(const NnFp16 value);
NnFp16 convertF32ToF16Impl(const float x);

#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
inline float convertF16ToF32Neon(const NnFp16 value) {
    __fp16 fp;
    std::memcpy(&fp, &value, sizeof(fp));
    return (float)fp;
}

inline NnFp16 convertF32ToF16Neon(const float x) {
    __fp16 h = x;
    return *(NnFp16 *)&h;
}

#define CONVERT_F16_TO_F32(value) convertF16ToF32Neon(value)
#define CONVERT_F32_TO_F16(value) convertF32ToF16Neon(value)
#endif

#if !defined(CONVERT_F16_TO_F32)
extern float f16ToF32Lookup[65536];

inline static float convertF16ToF32Lookup(const NnFp16 value) {
    return f16ToF32Lookup[value];
}

#define CONVERT_F16_TO_F32_LOOKUP
#define CONVERT_F16_TO_F32(value) convertF16ToF32Lookup(value)
#endif

#if !defined(CONVERT_F32_TO_F16)
#define CONVERT_F32_TO_F16(value) convertF32ToF16Impl(value)
#endif

#define Q40_BLOCK_SIZE 32
#define Q80_BLOCK_SIZE 32

enum NnFloatType {
    F_UNK = -1,
    F_32 = 0,
    F_16 = 1,
    F_Q40 = 2,
    F_Q80 = 3,
};

typedef struct {
    std::uint16_t d;
    std::uint8_t qs[Q40_BLOCK_SIZE / 2];
} NnBlockQ40;

typedef struct {
    std::uint16_t d;
    std::int8_t qs[Q80_BLOCK_SIZE];
} NnBlockQ80;

void initQuants();
void quantizeF32toQ80(const float *input, NnBlockQ80 *output, const NnSize k, const NnSize nThreads, const NnSize threadIndex);
void dequantizeQ80toF32(const NnBlockQ80 *input, float* output, const NnSize k, const NnSize nThreads, const NnSize threadIndex);
void quantizeF32toQ40(const float *x, NnBlockQ40 *output, const NnSize n, const NnSize nThreads, const NnSize threadIndex);
void dequantizeQ40toF32(const NnBlockQ40 *x, float *output, const NnSize n, const NnSize nThreads, const NnSize threadIndex);

const char *floatTypeToString(NnFloatType type);

#define SPLIT_THREADS(varStart, varEnd, rangeLen, nThreads, threadIndex) \
    const NnSize rangeSlice = rangeLen / nThreads; \
    const NnSize rangeRest = rangeLen % nThreads; \
    const NnSize varStart = threadIndex * rangeSlice + (threadIndex < rangeRest ? threadIndex : rangeRest); \
    const NnSize varEnd = varStart + rangeSlice + (threadIndex < rangeRest ? 1 : 0);

#endif