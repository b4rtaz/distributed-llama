#ifndef matmul_hpp
#define matmul_hpp

enum FloatType {
    F32 = 0,
    F16 = 1
};
long getFloatSize(FloatType type);

void matmul(FloatType type, float* output, float* input, char* weights, int n, int d);

class MatMulSlice {
public:
    int floatSize;
    int sliceCount;
    int d0;
    int n;
    long weights0Bytes;

    MatMulSlice(FloatType type, int sliceCount, int n, int d);
    long splitWeights(int sliceIndex, char* weights, char* weights0);
    long mergeOutputs(int sliceIndex, float* output, float* output0);
};

#endif
