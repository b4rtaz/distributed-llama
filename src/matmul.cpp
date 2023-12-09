#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "matmul.hpp"

struct matmul_thread_args {
    float* output;
    float* input;
    float* weights;
    int n;
    int ds;
    int de;
};

void* matmul_thread(void* arg) {
    matmul_thread_args* a = (matmul_thread_args*)arg;

    int i;
    for (i = a->ds; i < a->de; i++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            val += a->weights[i * a->n + j] * a->input[j];
        }
        a->output[i] = val;
    }
    return 0;
}

//     weights      input    output
//   ___________     ___      ___
//   |         |     | |      | |
// d |         | *   | |  = d | |
//   |_________|   n | |      |_|
//        n          |_|       1
//                    1
void matmul(float* output, float* input, float* weights, int n, int d) {
    // weights (d,n) @ input (n,1) -> output (d,1)
    
    int threadCount = 8;
    pthread_t threads[threadCount];
    matmul_thread_args args[threadCount];

    int i;
    for (i = 0; i < threadCount; i++) {
        matmul_thread_args* s = &args[i];
        s->output = output;
        s->input = input;
        s->weights = weights;
        s->n = n;
        s->ds = i * d / threadCount;
        s->de = (i + 1) * d / threadCount;
        int result = pthread_create(&threads[i], NULL, matmul_thread, (void*)s);
    }
    for (i = 0; i < threadCount; i++) {
        pthread_join(threads[i], NULL);
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

long MatMulSlice::splitWeights(int sliceIndex, float* weights, float* weights0) {
    long offset = this->d0 * sliceIndex * this->n;
    for (int i = 0; i < this->d0; i++) {
        for (int j = 0; j < this->n; j++) {
            weights0[i * this->n + j] = weights[offset + i * this->n + j];
        }
    }
    return offset;
}

long MatMulSlice::mergeOutputs(int sliceIndex, float* output, float* output0) {
    long offset = this->d0 * sliceIndex;
    for (int i = 0; i < this->d0; i++) {
        output[offset + i] = output0[i];
    }
    return offset;
}
