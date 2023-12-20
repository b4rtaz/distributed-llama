#include <cmath>
#include <cassert>
#include <sys/time.h>

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

long timeMs() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000LL + te.tv_usec / 1000;
}

unsigned int randomU32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float randomF32(unsigned long long *state) {
    // random float32 in <0,1)
    return (randomU32(state) >> 8) / 16777216.0f;
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    assert(size % 4 == 0);
    float ss = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t fsq;
    float32x4_t fs = vmovq_n_f32(0);
    for (int j = 0; j < size; j += 4) {
        fsq = vld1q_f32(&x[j]);
        fs = vmlaq_f32(fs, fsq, fsq);
    }
    ss = vaddvq_f32(fs);
#else
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
#endif
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

#if defined(__ARM_NEON)
    float32x4_t fw;
    float32x4_t fx;
    float32x4_t fss = vmovq_n_f32(ss);
    for (int j = 0; j < size; j += 4) {
        fw = vld1q_f32(&weight[j]);
        fx = vld1q_f32(&x[j]);
        fx = vmulq_f32(fx, fw);
        fx = vmulq_f32(fx, fss);
        vst1q_f32(&o[j], fx);
    }
#else
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
#endif
}

void softmax(float* x, int size) {
    float maxVal;
#if defined(__ARM_NEON)
    float32x4_t fs;
    float32x4_t fmaxv = vld1q_f32(&x[0]);
    for (int i = 4; i < size; i += 4) {
        fs = vld1q_f32(&x[i]);
        fmaxv = vmaxq_f32(fmaxv, fs);
    }
    maxVal = vmaxvq_f32(fmaxv);
#else
    // find max value (for numerical stability)
    maxVal = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > maxVal) {
            maxVal = x[i];
        }
    }
#endif
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float dotProduct(float* a, float* b, int size) {
    assert(size % 4 == 0);
#if defined(__ARM_NEON)
    float32x4_t fa;
    float32x4_t fb;
    float32x4_t fs = vmovq_n_f32(0);
    for (int i = 0; i < size; i += 4) {
        fa = vld1q_f32(&a[i]);
        fb = vld1q_f32(&b[i]);
        fs = vmlaq_f32(fs, fa, fb);
    }
    return vaddvq_f32(fs);
#else
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}
