#ifndef quants_hpp
#define quants_hpp

enum FloatType {
    F32 = 0,
    F16 = 1,
    Q40 = 2
};

#define QK40 32

typedef struct {
    uint16_t d; // delta
    uint8_t qs[QK40 / 2]; // nibbles / quants
} BlockQ40;

void initQuants();

int getNumbersPerBatch(FloatType type);
long getBatchBytes(FloatType type, int n, int d);
float convertF16ToF32(uint16_t value);

void dequantizeQ40Row(const BlockQ40* x, float* y, int k);

#endif
