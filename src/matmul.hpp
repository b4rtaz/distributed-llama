#ifndef matmul_hpp
#define matmul_hpp

void matmul(float* output, float* input, float* weights, int n, int d);

class MatMulSlice {
public:
    int sliceCount;
    int d0;
    int n;
    int weights0Length; // Number of floats in weights0

    MatMulSlice(int sliceCount, int n, int d);
    int splitWeights(int sliceIndex, float* weights, float* weights0);
    int mergeOutputs(int sliceIndex, float* output, float* output0);
};

#endif
