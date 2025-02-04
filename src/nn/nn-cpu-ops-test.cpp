#include "nn-cpu-ops.cpp"

// framework

void rand(float *o, const NnSize n, const NnSize seed) {
    srand(seed + 123456);
    for (NnSize i = 0; i < n; i++) {
        float v = (float)rand() / RAND_MAX;
        o[i] = v * 2.0f - 1.0f;
    }
}

void compare_F32(const char *name, const float *a, const float *b, const NnSize n, const float epsilon) {
    for (NnSize i = 0; i < n; i++) {
        float error = fabs(a[i] - b[i]);
        if (error > epsilon) {
            printf("❌ %s failed\n", name);
            for (NnSize j = i; j < i + 16 && j < n; j++)
                printf("   [%3d] %f != %f\n", j, a[j], b[j]);
            exit(1);
        }
    }
    printf("✅ %18s passed (%.3f %.3f %.3f %.3f...)\n", name, a[0], a[1], a[2], a[3]);
}

// tests

void testSplitThreads() {
    // <0; 32> across 3 threads
    {
        SPLIT_THREADS(a0Start, a0End, 32, 3, 0); // thread 0
        assert(a0Start == 0);
        assert(a0End == 11);
    }
    {
        SPLIT_THREADS(a1Start, a1End, 32, 3, 1); // thread 1
        assert(a1Start == 11);
        assert(a1End == 22);
    }
    {
        SPLIT_THREADS(a2Start, a2End, 32, 3, 2); // thread 2
        assert(a2Start == 22);
        assert(a2End == 32);
    }

    // <0; 4> across 8 threads
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 0); // thread 0
        assert(b0Start == 0);
        assert(b0End == 1);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 3); // thread 3
        assert(b0Start == 3);
        assert(b0End == 4);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 4); // thread 4
        assert(b0Start == 4); 
        assert(b0End == 4);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 7); // thread 7
        assert(b0Start == 4);
        assert(b0End == 4);
    }

    printf("✅ splitThreads\n");
}

// quantization
void testQuantization(const NnSize m) {
    std::vector<float> a(m * Q40_BLOCK_SIZE);
    std::vector<float> aTemp(m * Q40_BLOCK_SIZE);
    std::vector<NnBlockQ40> aQ40(m);
    std::vector<NnBlockQ80> aQ80(m);

    rand(a.data(), m * Q40_BLOCK_SIZE, m);

    quantizeF32toQ40(a.data(), aQ40.data(), m * Q40_BLOCK_SIZE, 1, 0);
    dequantizeQ40toF32(aQ40.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q40", a.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 0.12);

    quantizeF32toQ80(a.data(), aQ80.data(), m * Q80_BLOCK_SIZE, 1, 0);
    dequantizeQ80toF32(aQ80.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q80", a.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 0.01);
}

// rms
void testRms(const NnSize m) {
    const float epsilon = 0.00001;

    std::vector<float> x(m);
    rand(x.data(), m, m);

    const float y0 = rms_F32(x.data(), m, epsilon);

    float expectedValue = 1.616181f;
    compare_F32("rms_Q80", &y0, &expectedValue, 1, 0.001);
}

// rmsNorm
void testRmsNorm(const NnSize m) {
    std::vector<float> x(m);
    std::vector<NnBlockQ80> xQ80(m / Q80_BLOCK_SIZE);
    std::vector<float> w(m);
    std::vector<float> y(m);
    std::vector<float> yTemp(m);

    rand(x.data(), m, m);
    rand(w.data(), m, m * m);
    quantizeF32toQ80(x.data(), xQ80.data(), m, 1, 0);
    const float rms = rms_F32(x.data(), m, 1e-5f);

    rmsNorm_F32(y.data(), x.data(), rms, w.data(), m, 1, 0);
    rmsNorm_Q80_F32_F32(yTemp.data(), xQ80.data(), rms, w.data(), m, 1, 0);

    compare_F32("rmsNorm_Q80_F32_F32", y.data(), yTemp.data(), m, 0.01);
}

// a *= b
void testMul(const NnSize m) {
    const NnSize n = Q80_BLOCK_SIZE * m;

    std::vector<float> a0(n);
    std::vector<float> b0(n);

    std::vector<float> aQ(n);
    std::vector<NnBlockQ80> b1(n / Q80_BLOCK_SIZE);

    rand(a0.data(), n, m);
    rand(aQ.data(), n, m);
    rand(b0.data(), n, m);
    quantizeF32toQ80(b0.data(), b1.data(), n, 1, 0);

    mul_F32(a0.data(), b0.data(), n, 1, 0);
    mul_Q80_F32(aQ.data(), b1.data(), n, 1, 0);

    compare_F32("mul_Q80_F32", a0.data(), aQ.data(), n, 0.005);
}

// y += x
void testAdd(const NnSize m) {
    const NnSize n = Q80_BLOCK_SIZE * m;

    std::vector<float> y(n);
    std::vector<float> yTemp(n);
    std::vector<float> x(n);
    std::vector<NnBlockQ80> xQ80(n / Q80_BLOCK_SIZE);

    rand(y.data(), n, m);
    rand(yTemp.data(), n, m);
    rand(x.data(), n, m);
    quantizeF32toQ80(x.data(), xQ80.data(), n, 1, 0);

    add_F32(y.data(), x.data(), n, 1, 0);
    add_Q80_F32(yTemp.data(), xQ80.data(), n, 1, 0);

    compare_F32("add_Q80_F32", y.data(), yTemp.data(), n, 0.01);
}

// matmul
void testMatmul_F32_Q40_F32(const NnSize m = 2) {
    const NnSize n = Q80_BLOCK_SIZE * m;
    const NnSize d = Q80_BLOCK_SIZE * m;

    std::vector<float> x(n);
    std::vector<float> w(n * d);
    std::vector<float> o(d);
    std::vector<float> oTemp(d);
    std::vector<NnBlockQ80> xQ80(n / Q80_BLOCK_SIZE);
    std::vector<NnBlockQ80> oQ80(d / Q80_BLOCK_SIZE);
    std::vector<NnBlockQ40> wQ40((n * d) / Q40_BLOCK_SIZE);

    rand(x.data(), n, m);
    rand(w.data(), n * d, m);
    quantizeF32toQ40(w.data(), wQ40.data(), n * d, 1, 0);
    quantizeF32toQ80(x.data(), xQ80.data(), n, 1, 0);

    matmul_F32_F32_F32(o.data(), x.data(), w.data(), n, d, 1, 0);
    matmul_F32_Q40_F32(oTemp.data(), x.data(), wQ40.data(), n, d, 1, 0);
    compare_F32("matmul_F32_Q40_F32", o.data(), oTemp.data(), d, 3.0f);

    matmul_F32_Q40_Q80(oQ80.data(), x.data(), wQ40.data(), n, d, 1, 0);
    dequantizeQ80toF32(oQ80.data(), oTemp.data(), d, 1, 0);
    compare_F32("matmul_F32_Q40_Q80", o.data(), oTemp.data(), d, 3.0f);

    matmul_Q80_Q40_F32(oTemp.data(), xQ80.data(), wQ40.data(), n, d, 1, 0);
    compare_F32("matmul_Q80_Q40_F32", o.data(), oTemp.data(), d, 2.8f);
}

int main() {
    testSplitThreads();
    testQuantization(32);
    testQuantization(2);
    testQuantization(1);
    testRms(128);
    testRmsNorm(128);
    testMul(32);
    testMul(2);
    testMul(1);
    testAdd(32);
    testAdd(2);
    testAdd(1);
    testMatmul_F32_Q40_F32(32);
    testMatmul_F32_Q40_F32(2);
    testMatmul_F32_Q40_F32(1);
    return 0;
}
