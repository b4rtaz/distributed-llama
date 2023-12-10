#ifndef quants_hpp
#define quants_hpp

enum FloatType {
    F32 = 0,
    F16 = 1,
};

void initQuants();

long getFloatBytes(FloatType type);
float convertF16ToF32(uint16_t value);

#endif
