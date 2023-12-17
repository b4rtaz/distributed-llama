#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <pthread.h>
#include "quants.hpp"
#include "matmul.hpp"

#define NEON 1

#if NEON
    #include <arm_neon.h>
#endif

struct MatmulThreadInfo {
    pthread_t handler;
    float* output;
    float* input;
    char* weights;
    FloatType type;
    int n;
    int ds;
    int de;
};

void matmulF32(MatmulThreadInfo* a) {
    float* w = (float*)a->weights;
    int d;

#if NEON
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (d = a->ds; d < a->de; d++) {
        z = vmovq_n_f32(0);
        for (int j = 0; j < a->n; j += 4) {
            q = vld1q_f32(&a->input[j]);
            p = vld1q_f32(&w[d * a->n + j]);
            z = vfmaq_f32(z, q, p);
        }
        a->output[d] = vaddvq_f32(z);
    }
#else
    for (d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            val += w[d * a->n + j] * a->input[j];
        }
        a->output[d] = val;
    }
#endif
}

void matmulF16(MatmulThreadInfo* a) {
    uint16_t* w = (uint16_t*)a->weights;
    int d;
    for (d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            float ww = convertF16ToF32(w[d * a->n + j]);
            val += ww * a->input[j];
        }
        a->output[d] = val;
    }
}

void matmulQ40(MatmulThreadInfo* a) {
    const int blocksPerRow = 8;
    const int k = QK40 * blocksPerRow;
    BlockQ40* w = (BlockQ40*)a->weights;
    assert(a->n % k == 0);
    int n = a->n / k;
    float group[k];

#if NEON
    assert(k % 16 == 0);
    float32x4_t a0;
    float32x4_t b0;
    float32x4_t u;
    for (int d = a->ds; d < a->de; d++) {
        u = vmovq_n_f32(0);
        for (int j = 0; j < n; j++) {
            dequantizeQ40Row(&w[d * n * blocksPerRow + j * blocksPerRow], group, k);
            for (int z = 0; z < k; z += 4) {
                a0 = vld1q_f32(&a->input[j * k + z]);
                b0 = vld1q_f32(&group[z]);
                u = vfmaq_f32(u, a0, b0);
            }
        }
        a->output[d] = vaddvq_f32(u);
    }
#else
    for (int d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            dequantizeQ40Row(&w[d * n * blocksPerRow + j * blocksPerRow], group, k);
            for (int z = 0; z < k; z++) {
                val += group[z] * a->input[j * k + z];
            }
        }
        a->output[d] = val;
    }
#endif
}

void* matmulThread(void* arg) {
    MatmulThreadInfo* a = (MatmulThreadInfo*)arg;
    switch (a->type)
    {
        case F32:
            matmulF32(a);
            break;
        case F16:
            matmulF16(a);
            break;
        case Q40:
            matmulQ40(a);
            break;
        default:
            printf("Unknown float type %d\n", a->type);
            exit(EXIT_FAILURE);
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
void matmul(FloatType type, int nThread, float* output, float* input, char* weights, int n, int d) {
    MatmulThreadInfo args[nThread];

    int i;
    for (i = 0; i < nThread; i++) {
        MatmulThreadInfo* s = &args[i];
        s->output = output;
        s->input = input;
        s->weights = weights;
        s->type = type;
        s->n = n;
        s->ds = i * d / nThread;
        s->de = (i + 1) * d / nThread;
        int result = pthread_create(&args[i].handler, NULL, matmulThread, (void*)s);
    }
    for (i = 0; i < nThread; i++) {
        pthread_join(args[i].handler, NULL);
    }
}

MatMulSlice::MatMulSlice(FloatType type, int sliceCount, int n, int d) {
    assert(d % sliceCount == 0);

    this->type = type;
    this->sliceCount = sliceCount;
    this->d0 = d / sliceCount;
    this->n = n;
    this->weights0Bytes = getBatchBytes(type, this->n, this->d0);
}

long MatMulSlice::splitWeights(int sliceIndex, char* weights, char* weights0) {
    int numbersPerBatch = getNumbersPerBatch(this->type);
    int batchBytes = getBatchBytes(this->type, numbersPerBatch, 1);

    int n = this->n / numbersPerBatch;
    long offset = this->d0 * sliceIndex * n * batchBytes;

    for (int d = 0; d < this->d0; d++) {
        for (int j = 0; j < n; j++) {
            long o = (d * n + j) * batchBytes;

            memcpy(weights0 + o, weights + offset + o, batchBytes);
        }
    }
    return offset; // offset in bytes
}

long MatMulSlice::mergeOutputs(int sliceIndex, float* output, float* output0) {
    long offset = this->d0 * sliceIndex;
    for (int i = 0; i < this->d0; i++) {
        output[offset + i] = output0[i];
    }
    return offset; // offset in floats
}
