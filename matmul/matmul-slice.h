#include <stdio.h>
#include <stdlib.h>

struct matmul_slice {
    int slices;
    int d0;
    int n;
    int weights0Length; // Number of floats in weights0
};

void matmul_slice_new(
    struct matmul_slice* slice,
    int slices,
    int n,
    int d
) {
    if (d % slices != 0) {
        printf("d=%d must be divisible by slices=%d\n", d, slices);
        exit(1);
    }
    slice->slices = slices;
    slice->d0 = d / slices;
    slice->n = n;
    slice->weights0Length = slice->d0 * slice->n;
}

int matmul_slice_split_weights(
    struct matmul_slice* slice,
    int sliceIndex,
    float* weights,
    float* weights0
) {
    int offset = slice->d0 * sliceIndex * slice->n;
    for (int i = 0; i < slice->d0; i++) {
        for (int j = 0; j < slice->n; j++) {
            weights0[i * slice->n + j] = weights[offset + i * slice->n + j];
        }
    }
    return offset;
}

int matmul_slice_merge_output(
    struct matmul_slice* slice,
    int sliceIndex,
    float* output,
    float* output0
) {
    int offset = slice->d0 * sliceIndex;
    for (int i = 0; i < slice->d0; i++) {
        output[offset + i] = output0[i];
    }
    return offset;
}
