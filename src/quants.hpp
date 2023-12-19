#ifndef quants_hpp
#define quants_hpp

enum FloatType {
    F32 = 0,
    F16 = 1,
    Q40 = 2
};

#define QK40 32
#define QK80 32

typedef struct {
    uint16_t d; // delta
    uint8_t qs[QK40 / 2]; // nibbles / quants
} BlockQ40;

typedef struct {
    uint16_t d; // delta
    int8_t  qs[QK80]; // quants
} BlockQ80;

void initQuants();

int getNumbersPerBatch(FloatType type);
long getBatchBytes(FloatType type, int n, int d);
float convertF16ToF32(uint16_t value);

void dequantizeQ40Row(const BlockQ40* x, float* y, int k);
void quantizeQ80Row(float* x, BlockQ80* y, int k);

#endif
