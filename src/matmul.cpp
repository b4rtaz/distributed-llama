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
    void* input;
    void* weights;
    FloatType type;
    int n;
    int ds;
    int de;
};

void matmulF32(MatmulThreadInfo* a) {
    const float* input = (float*)a->input;
    float* w = (float*)a->weights;
    int d;

#if NEON
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (d = a->ds; d < a->de; d++) {
        z = vmovq_n_f32(0);
        for (int j = 0; j < a->n; j += 4) {
            q = vld1q_f32(&input[j]);
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
    const float* input = (float*)a->input;
    uint16_t* w = (uint16_t*)a->weights;
    int d;
    for (d = a->ds; d < a->de; d++) {
        float val = 0.0f;
        for (int j = 0; j < a->n; j++) {
            float ww = convertF16ToF32(w[d * a->n + j]);
            val += ww * input[j];
        }
        a->output[d] = val;
    }
}

void matmulQ40vQ80(MatmulThreadInfo* a) {
    const BlockQ40* w = (BlockQ40*)a->weights;
    const BlockQ80* input = (BlockQ80*)a->input;
    assert(a->n % QK40 == 0);
    const int n = a->n / QK40;

#if NEON
    float32x4_t sumv0;
    for (int d = a->ds; d < a->de; d++) {
        sumv0 = vmovq_n_f32(0);
        for (int j = 0; j < n; j++) {
            const BlockQ40* x0 = &w[d * n + j];
            const BlockQ80* y0 = &input[j];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(x0->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));

            // sub 8
            const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
            const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);

            // load y
            const int8x16_t v1_0l = vld1q_s8(y0->qs);
            const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p_0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), convertF16ToF32(x0->d) * convertF16ToF32(y0->d));
#else
        const int16x8_t pl0l = vmull_s8(vget_low_s8 (v0_0ls), vget_low_s8 (v1_0l));
        const int16x8_t pl0h = vmull_s8(vget_high_s8(v0_0ls), vget_high_s8(v1_0l));
        const int16x8_t ph0l = vmull_s8(vget_low_s8 (v0_0hs), vget_low_s8 (v1_0h));
        const int16x8_t ph0h = vmull_s8(vget_high_s8(v0_0hs), vget_high_s8(v1_0h));

        const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
        const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), convertF16ToF32(x0->d)*convertF16ToF32(y0->d));
#endif
        }
        a->output[d] = vaddvq_f32(sumv0);
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
            matmulQ40vQ80(a);
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
void matmul(FloatType type, int nThread, float* output, float* input, void* weights, int n, int d) {
    MatmulThreadInfo args[nThread];

    if (type == Q40) {
        BlockQ80* bq80 = new BlockQ80[n / QK80];
        quantizeQ80Row(input, bq80, n);
        input = (float*)bq80;
    }

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

    if (type == Q40) {
        delete[] input;
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
