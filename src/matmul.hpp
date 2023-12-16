#include "quants.hpp"

#ifndef matmul_hpp
#define matmul_hpp

void matmul(FloatType type, float* output, float* input, char* weights, int n, int d);

class MatMulSlice {
public:
    FloatType type;
    int sliceCount;
    int d0;
    int n;
    size_t weights0Bytes;

    MatMulSlice(FloatType type, int sliceCount, int n, int d);
    long splitWeights(int sliceIndex, char* weights, char* weights0);
    long mergeOutputs(int sliceIndex, float* output, float* output0);
};

#endif
