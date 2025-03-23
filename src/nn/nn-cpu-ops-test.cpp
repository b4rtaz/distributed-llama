#include "nn-cpu-ops.cpp"
#include <vector>

// framework

void rand(float *o, const NnUint n, const NnUint seed) {
    srand(seed + 123456);
    for (NnUint i = 0; i < n; i++) {
        float v = (float)(rand() / RAND_MAX);
        o[i] = v * 2.0f - 1.0f;
    }
}

void compare_F32(const char *name, const float *a, const float *b, const NnUint n, const float epsilon) {
    for (NnUint i = 0; i < n; i++) {
        float error = fabs(a[i] - b[i]);
        if (error > epsilon) {
            printf("❌ %s failed\n", name);
            for (NnUint j = i; j < i + 16 && j < n; j++)
                printf("   [%3d] %f != %f\n", j, a[j], b[j]);
            exit(1);
        }
    }
    printf("✅ %24s passed\n", name);
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

    printf("✅ %24s passed\n", "splitThreads");
}

void testConvertF32toF16() {
    float x[] = {0.0f, 0.25f, 0.3456f, 1.0f};
    for (NnUint i = 0; i < sizeof(x) / sizeof(float); i++) {
        NnFp16 f16 = CONVERT_F32_TO_F16(x[i]);
        float f32 = CONVERT_F16_TO_F32(f16);
        compare_F32("convertF32toF16", &x[i], &f32, 1, 0.0005);
    }
}

// quantization
void testQuantization(const NnUint m) {
    std::vector<float> a(m * Q40_BLOCK_SIZE);
    std::vector<float> aTemp(m * Q40_BLOCK_SIZE);
    std::vector<NnBlockQ40> aQ40(m);
    std::vector<NnBlockQ80> aQ80(m);

    rand(a.data(), m * Q40_BLOCK_SIZE, m);

    quantizeF32toQ40(a.data(), aQ40.data(), m * Q40_BLOCK_SIZE, 1, 0);
    dequantizeQ40toF32(aQ40.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q40", a.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 0.13);

    quantizeF32toQ80(a.data(), aQ80.data(), m * Q80_BLOCK_SIZE, 1, 0);
    dequantizeQ80toF32(aQ80.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q80", a.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 0.01);
}

// invRms
void testInvRms() {
    const float epsilon = 0.00001;

    std::vector<float> x(8);
    x[0] = 0.1f;
    x[1] = 0.3f;
    x[2] = 0.2f;
    x[3] = 0.4f;
    x[4] = 0.6f;
    x[5] = 0.5f;
    x[6] = 0.0f;
    x[7] = 0.8f;

    const float y0 = invRms_F32(x.data(), 8, epsilon);
    float ev0 = 1.0f / 0.4402f;
    compare_F32("rms_F32", &y0, &ev0, 1, 0.001f);
}

// rmsNorm
void testRmsNorm(const NnUint m) {
    std::vector<float> x(m);
    std::vector<NnBlockQ80> xQ80(m / Q80_BLOCK_SIZE);
    std::vector<float> w(m);
    std::vector<float> y(m);
    std::vector<float> yTemp(m);

    rand(x.data(), m, m);
    rand(w.data(), m, m * m);
    quantizeF32toQ80(x.data(), xQ80.data(), m, 1, 0);
    const float rms = invRms_F32(x.data(), m, 1e-5f);

    rmsNorm_F32(y.data(), x.data(), rms, w.data(), m, 1, 0);
    rmsNorm_Q80_F32_F32(yTemp.data(), xQ80.data(), rms, w.data(), m, 1, 0);

    compare_F32("rmsNorm_Q80_F32_F32", y.data(), yTemp.data(), m, 0.01);
}

// a *= b
void testMul(const NnUint m) {
    const NnUint n = Q80_BLOCK_SIZE * m;

    std::vector<float> a0(n);
    std::vector<float> b0(n);

    std::vector<float> aQ(n);
    std::vector<NnBlockQ80> b1(n / Q80_BLOCK_SIZE);

    rand(a0.data(), n, m);
    rand(aQ.data(), n, m);
    rand(b0.data(), n, m);
    quantizeF32toQ80(b0.data(), b1.data(), n, 1, 0);

    mul_F32(a0.data(), a0.data(), b0.data(), n, 1, 0);
    mul_Q80_F32(aQ.data(), aQ.data(), b1.data(), n, 1, 0);

    compare_F32("mul_Q80_F32", a0.data(), aQ.data(), n, 0.005);
}

// y += x
void testAdd(const NnUint m) {
    const NnUint n = Q80_BLOCK_SIZE * m;

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

void testSoftmax() {
    std::vector<float> y(8);
    for (NnUint i = 0; i < 8; i++)
        y[i] = i / 8.0f;

    softmax_F32(y.data(), 8);

    float expectedOutput[8] = {
        0.077399f,
        0.087780f,
        0.099500f,
        0.112761f,
        0.127778f,
        0.144793f,
        0.164072f,
        0.185917f
    };
    compare_F32("softmax_F32", y.data(), expectedOutput, 8, 0.001);
}

void testSilu() {
    std::vector<float> y(8);
    for (NnUint i = 0; i < 8; i++)
        y[i] = i / 8.0f;

    silu_F32(y.data(), 8, 1, 0);

    float expectedOutput[8] = {
        0.000000f,
        0.066401f,
        0.140544f,
        0.222250f,
        0.311233f,
        0.407116f,
        0.509461f,
        0.617802f
    };
    compare_F32("silu_F32", y.data(), expectedOutput, 8, 0.001);
}

// matmul
void testMatmul_F32_Q40_F32(const NnUint m = 2) {
    const NnUint n = Q80_BLOCK_SIZE * m;
    const NnUint d = Q80_BLOCK_SIZE * m;

    std::vector<float> x(n);
    std::vector<float> w(n * d);
    std::vector<float> o(d);
    std::vector<float> oTemp(d);
    std::vector<NnBlockQ80> xQ80(n / Q80_BLOCK_SIZE);
    std::vector<NnBlockQ40> wQ40((n * d) / Q40_BLOCK_SIZE);

    rand(x.data(), n, m);
    rand(w.data(), n * d, m);
    quantizeF32toQ40(w.data(), wQ40.data(), n * d, 1, 0);
    quantizeF32toQ80(x.data(), xQ80.data(), n, 1, 0);

    matmul_F32_F32_F32(o.data(), x.data(), w.data(), n, d, 1, 0);

    matmul_Q80_Q40_F32(oTemp.data(), xQ80.data(), wQ40.data(), n, d, 1, 0);
    compare_F32("matmul_Q80_Q40_F32", o.data(), oTemp.data(), d, 4.0f);
}

void testLlamafileSgemm() {
    const NnUint batchSize = 8;
    const NnUint n = 256;
    const NnUint d = 128;

    std::vector<float> x(n * batchSize);
    std::vector<NnBlockQ80> xQ((n * batchSize) / Q80_BLOCK_SIZE);
    std::vector<float> w(n * d);
    std::vector<NnBlockQ40> wQ((n * d) / Q40_BLOCK_SIZE);
    std::vector<float> o(d * batchSize);
    std::vector<float> oTemp(d * batchSize);

    rand(x.data(), n * batchSize, 12345);
    rand(w.data(), n * d, 23456);

    quantizeF32toQ80(x.data(), xQ.data(), n * batchSize, 1, 0);
    quantizeF32toQ40(w.data(), wQ.data(), n * d, 1, 0);

    // f32

    for (NnUint i = 0; i < batchSize; i++) {
        matmul_F32_F32_F32(o.data() + i * d, x.data() + i * n, w.data(), n, d, 1, 0);
    }

    assert(llamafile_sgemm(
        d, batchSize, n,
        w.data(), n,
        x.data(), n,
        oTemp.data(), d,
        0, 1, 0,
        F_32, F_32, F_32
    ));

    compare_F32("llamafileSgemm_F32", o.data(), oTemp.data(), d * batchSize, 0.01f);

    // q40ᵀ * q80

    assert(llamafile_sgemm(
        d, batchSize, n / Q80_BLOCK_SIZE,
        wQ.data(), n / Q80_BLOCK_SIZE,
        xQ.data(), n / Q80_BLOCK_SIZE,
        oTemp.data(), d,
        0, 1, 0,
        F_Q40, F_Q80, F_32
    ));

    compare_F32("llamafileSgemm_Q80_Q40", o.data(), oTemp.data(), d * batchSize, 1.5f);
}

int main() {
    initQuants();

    printCpuInstructionSet();
    testSplitThreads();
    testConvertF32toF16();
    testQuantization(32);
    testQuantization(2);
    testQuantization(1);
    testInvRms();
    testRmsNorm(128);
    testMul(32);
    testMul(2);
    testMul(1);
    testAdd(32);
    testAdd(2);
    testAdd(1);
    testSoftmax();
    testSilu();
    testMatmul_F32_Q40_F32(32);
    testMatmul_F32_Q40_F32(2);
    testMatmul_F32_Q40_F32(1);
    testLlamafileSgemm();
    return 0;
}
