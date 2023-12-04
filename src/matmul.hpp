void matmul(float* output, float* input, float* weights, int n, int d);

class MatMulSlice {
public:
    int slices;
    int d0;
    int n;
    int weights0Length; // Number of floats in weights0

    MatMulSlice(int slices, int n, int d);
    int splitWeights(int sliceIndex, float* weights, float* weights0);
    int mergeOutputs(int sliceIndex, float* output, float* output0);
};
