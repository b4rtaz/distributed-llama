#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matmul.h"

void read_floats(float* target, int n, char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) exit(-1);
    fread(target, sizeof(float), n, f);
    fclose(f);
    printf("Read %d floats from %s\n", n, path);
}

int main() {
    int n = 4096;
    int d = 4096;

    float* input = calloc(n, sizeof(float));
    float* weights = calloc(n * d, sizeof(float));
    float* output = calloc(d, sizeof(float));
    float* expectedOutput = calloc(d, sizeof(float));
    read_floats(input, n, "matmul/test0-input.data");
    read_floats(weights, n * d, "matmul/test0-weights.data");
    memset(output, 0, d * sizeof(float));
    read_floats(expectedOutput, d, "matmul/test0-output.data");

    matmul(output, input, weights, n, d);

    int r = memcmp(output, expectedOutput, n * sizeof(float));
    free(input);
    free(weights);
    free(output);
    free(expectedOutput);

    printf("r = %d\n", r);
    if (r == 0) {
        printf("✅ Test passed!\n");
        return 0;
    } else {
        printf("❌ Test failed!\n");
        return -1;
    }
}
