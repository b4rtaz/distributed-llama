#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matmul.hpp"

int n = 4096;
int d = 4096;
float* input; // (4096,1)
float* weights; // (4096x4096)
float* output; // (4096,1)
float* expectedOutput;

void test0_default() {
    matmul(output, input, weights, n, d);
}

void test0_distr() {
    int slices = 8;

    MatMulSlice* slice = new MatMulSlice(slices, n, d);

    for (int s = 0; s < slices; s++) {
        float* weights0 = new float[slice->weights0Length];

        long weightsOffset = slice->splitWeights(s, weights, weights0);

        float* output0 = new float[slice->d0];
        matmul(output0, input, weights0, slice->n, slice->d0);

        long outputOffset = slice->mergeOutputs(s, output, output0);

        delete[] weights0;
        delete[] output0;

        printf("weights <%8ld, %8ld> output <%4ld, %4ld>\n",
            weightsOffset,
            weightsOffset + slice->weights0Length,
            outputOffset,
            outputOffset + slice->d0);
    }

    delete slice;
}

void compareOrFail(const char *name) {
    int ix = -1;
    for (int i = 0; i < d; i++) {
        if (output[i] != expectedOutput[i]) {
            ix = i;
            break;
        }
    }
    if (ix < 0) {
        printf("[%s] ✅\n", name);
    } else {
        printf("[%s] ❌ ix=%d\n", name, ix);
        printf("%f != %f\n", output[ix], expectedOutput[ix]);
        exit(-1);
    }
}

void readData(float* target, int n, const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) exit(-1);
    fread(target, sizeof(float), n, f);
    fclose(f);
    printf("Read %d floats from %s\n", n, path);
}

int main() {
    input = new float[n];
    weights = new float[n * d];
    output = new float[d];
    expectedOutput = new float[d];
    readData(input, n, "test-data/matmul-input.data");
    readData(weights, n * d, "test-data/matmul-weights.data");
    readData(expectedOutput, d, "test-data/matmul-output.data");

    memset(output, 0, d * sizeof(float));
    test0_default();
    compareOrFail("default");

    memset(output, 0, d * sizeof(float));
    test0_distr();
    compareOrFail("distr");

    delete[] input;
    delete[] weights;
    delete[] output;
    delete[] expectedOutput;
    return 0;
}
