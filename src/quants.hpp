#ifndef QUANTS_HPP
#define QUANTS_HPP

#include <cstdint>

enum FloatType { // --> 量化类型
    FUNK = -1,
    F32 = 0,
    F16 = 1,
    Q40 = 2,
    Q80 = 3
};
// enum --> 枚举类型

#define QK40 32
#define QK80 32

typedef struct {
    uint16_t d; // delta
    uint8_t qs[QK40 / 2]; // nibbles / quants --> qs[16] --> 16位无符号整数
} BlockQ40;

typedef struct {
    uint16_t d; // delta
    int8_t  qs[QK80]; // quants --> qs[32] --> 32位有符号整数
} BlockQ80;

void initQuants();

int getNumbersPerBatch(FloatType type);// F32 | F16 -> 1, Q40 | Q80 -> 32
long getBatchBytes(FloatType type, int n, int d);
float convertF16ToF32(uint16_t value);

void dequantizeQ40Row(const BlockQ40* x, float* y, int k);
void quantizeQ80Row(float* input, BlockQ80* output, int k, unsigned int nThreads, unsigned int threadIndex);
void dequantizeQ80Row(const BlockQ80* input, float* output, int k, unsigned int nThreads, unsigned int threadIndex);

#endif
