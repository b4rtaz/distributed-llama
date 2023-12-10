#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "matmul.hpp"

long getFloatSize(FloatType type) {
    switch (type) {
        case F32:
            return sizeof(float);
        case F16:
            return sizeof(uint16_t);
        default:
            printf("Unknown float type %d\n", type);
            exit(1);
    }
}

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
    int i;
    for (i = a->ds; i < a->de; i++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            val += w[i * a->n + j] * a->input[j];
        }
        a->output[i] = val;
    }
}

// https://gist.github.com/rygorous/2144712
float fp16To32(uint16_t value) {
    union Fp32
    {
        uint32_t u;
        float f;
    };

    const Fp32 magic = { (254U - 15U) << 23 };
    const Fp32 was_infnan = { (127U + 16U) << 23 };
    Fp32 out;

    out.u = (value & 0x7FFFU) << 13;
    out.f *= magic.f;
    if (out.f >= was_infnan.f) {
        out.u |= 255U << 23;
    }
    out.u |= (value & 0x8000U) << 16;
    return out.f;
}

bool f16Init = false;
float f16Table[65536];

void initF16Table() {
    for (unsigned int i = 0; i < 65536; i++) {
        f16Table[(uint16_t)i] = fp16To32(i);
    }
}

void matmulF16(MatmulThreadInfo* a) {
    if (!f16Init) {
        initF16Table();
        f16Init = true;
    }

    uint16_t* w = (uint16_t*)a->weights;
    int i;
    for (i = a->ds; i < a->de; i++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            float ww = f16Table[w[i * a->n + j]];
            val += ww * a->input[j];
        }
        a->output[i] = val;
    }
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
        default:
            printf("Unknown float type %d\n", a->type);
            exit(1);
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
void matmul(FloatType type, float* output, float* input, char* weights, int n, int d) {
    int threadCount = 8;
    MatmulThreadInfo args[threadCount];

    int i;
    for (i = 0; i < threadCount; i++) {
        MatmulThreadInfo* s = &args[i];
        s->output = output;
        s->input = input;
        s->weights = weights;
        s->type = type;
        s->n = n;
        s->ds = i * d / threadCount;
        s->de = (i + 1) * d / threadCount;
        int result = pthread_create(&args[i].handler, NULL, matmulThread, (void*)s);
    }
    for (i = 0; i < threadCount; i++) {
        pthread_join(args[i].handler, NULL);
    }
}

MatMulSlice::MatMulSlice(FloatType type, int sliceCount, int n, int d) {
    if (d % sliceCount != 0) {
        printf("d=%d must be divisible by sliceCount=%d\n", d, sliceCount);
        exit(1);
    }
    this->floatSize = getFloatSize(type);
    this->sliceCount = sliceCount;
    this->d0 = d / sliceCount;
    this->n = n;
    this->weights0Bytes = this->d0 * this->n * sizeof(float);
}

long MatMulSlice::splitWeights(int sliceIndex, char* weights, char* weights0) {
    long offset = (this->d0 * sliceIndex * this->n) * floatSize;
    for (int i = 0; i < this->d0; i++) {
        for (int j = 0; j < this->n; j++) {
            for (int b = 0; b < floatSize; b++) {
                weights0[(i * this->n + j) * floatSize + b] = weights[offset + (i * this->n + j) * floatSize + b];
            }
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
