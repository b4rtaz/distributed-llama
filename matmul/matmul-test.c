#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matmul.h"
#include "matmul-slice.h"

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

    struct matmul_slice slice;
    matmul_slice_new(&slice, slices, n, d);

    for (int s = 0; s < slices; s++) {
        float* weights0 = calloc(slice.weights0Length, sizeof(float));

        int weightsOffset = matmul_slice_split_weights(&slice, s, weights, weights0);

        float* output0 = calloc(slice.d0, sizeof(float));
        matmul(output0, input, weights0, n, slice.d0);

        int outputOffset = matmul_slice_merge_output(&slice, s, output, output0);

        free(weights0);
        free(output0);

        printf("weights <%8d, %8d> output <%4d, %4d>\n",
            weightsOffset,
            weightsOffset + slice.weights0Length,
            outputOffset,
            outputOffset + slice.d0);
    }
}

void compareOrFail(char *name) {
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

void readData(float* target, int n, char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) exit(-1);
    fread(target, sizeof(float), n, f);
    fclose(f);
    printf("Read %d floats from %s\n", n, path);
}

int main() {
    input = calloc(n, sizeof(float));
    weights = calloc(n * d, sizeof(float));
    output = calloc(d, sizeof(float));
    expectedOutput = calloc(d, sizeof(float));
    readData(input, n, "matmul/test0-input.data");
    readData(weights, n * d, "matmul/test0-weights.data");
    readData(expectedOutput, d, "matmul/test0-output.data");

    memset(output, 0, d * sizeof(float));
    test0_default();
    compareOrFail("default");

    memset(output, 0, d * sizeof(float));
    test0_distr();
    compareOrFail("distr");

    free(input);
    free(weights);
    free(output);
    free(expectedOutput);
    return 0;
}
