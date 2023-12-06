#include <stdio.h>
#include <stdlib.h>
#include "matmul.hpp"

//     weights      input    output
//   ___________     ___      ___
//   |         |     | |      | |
// d |         | *   | |  = d | |
//   |_________|   n | |      |_|
//        n          |_|       1
//                    1
void matmul(float* output, float* input, float* weights, int n, int d) {
    // weights (d,n) @ input (n,1) -> output (d,1)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += weights[i * n + j] * input[j];
        }
        output[i] = val;
    }
}

MatMulSlice::MatMulSlice(int sliceCount, int n, int d) {
    if (d % sliceCount != 0) {
        printf("d=%d must be divisible by sliceCount=%d\n", d, sliceCount);
        exit(1);
    }
    this->sliceCount = sliceCount;
    this->d0 = d / sliceCount;
    this->n = n;
    this->weights0Length = this->d0 * this->n;
}

int MatMulSlice::splitWeights(int sliceIndex, float* weights, float* weights0) {
    int offset = this->d0 * sliceIndex * this->n;
    for (int i = 0; i < this->d0; i++) {
        for (int j = 0; j < this->n; j++) {
            weights0[i * this->n + j] = weights[offset + i * this->n + j];
        }
    }
    return offset;
}

int MatMulSlice::mergeOutputs(int sliceIndex, float* output, float* output0) {
    int offset = this->d0 * sliceIndex;
    for (int i = 0; i < this->d0; i++) {
        output[offset + i] = output0[i];
    }
    return offset;
}
