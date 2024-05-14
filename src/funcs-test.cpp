#include "funcs.hpp"
#include "utils.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

void testRms() {
    float x[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float r = rms(x, 8);
    if (fabs(r - 1.980256) > 0.001) {
        printf("❌ rms() = %f\n", r);
        exit(EXIT_FAILURE);
    }
    printf("✅ rms\n");
}

void testMatmulQ80() {
    const int n = 512;
    const int d = 256;
    unsigned long long state = 88888888L;
    float x[n];
    float w[n * d];
    float y[d];
    float yQ0[d];
    float yQ1[d];
    int i;
    for (i = 0; i < n; i++) x[i] = randomF32(&state) / 127.0f;
    for (i = 0; i < n * d; i++) w[i] = randomF32(&state) / 127.0f;

    char* xQ = new char[getBatchBytes(Q80, n, 1)];
    char* wQ = new char[getBatchBytes(Q80, n, d)];
    quantizeQ80Row(x, (BlockQ80*)xQ, n, 1, 0);
    quantizeQ80Row(w, (BlockQ80*)wQ, n * d, 1, 0);

    matmul(F32, F32, y, x, w, n, d, 1, 0);
    matmul(Q80, F32, yQ0, x, wQ, n, d, 1, 0);
    matmul(Q80, Q80, yQ1, xQ, wQ, n, d, 1, 0);

    for (i = 0; i < d; i++) {
        float diff = fabs(y[i] - yQ0[i]);
        if (diff > 0.001) {
            printf("❌ matmulQ80() ix=%d %f != %f diff=%f\n", i, y[i], yQ0[i], diff);
            exit(EXIT_FAILURE);
        }
    }
    printf("✅ matmulQ80\n");

    for (i = 0; i < d; i++) {
        float diff = fabs(y[i] - yQ1[i]);
        if (diff > 0.001) {
            printf("❌ matmulQ80vQ80() ix=%d %f != %f diff=%f\n", i, y[i], yQ1[i], diff);
            exit(EXIT_FAILURE);
        }
    }
    printf("✅ matmulQ80vQ80\n");

    delete[] xQ;
    delete[] wQ;
}

void testAdd() {
    const int n = 16;
    float a[n];
    float b[n];

    for (int nThreads = 1; nThreads < 8; nThreads++) {
        for (int i = 0; i < n; i++) {
            a[i] = (float)-i;
            b[i] = (float)i;
        }

        for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
            add(a, b, n, nThreads, threadIndex);
        }

        for (int i = 0; i < n; i++) {
            if (fabs(a[i]) > 0.001) {
                printf("❌ add() = %f (nThreads=%d)\n", a[i], nThreads);
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("✅ add\n");
}

void assertInt(int a, int b) {
    if (a != b) {
        printf("❌ %d != %d\n", a, b);
        exit(EXIT_FAILURE);
    }
}

void testSplitRangeToThreads() {
    // <0; 32> across 3 threads
    {
        SPLIT_RANGE_TO_THREADS(a0Start, a0End, 0, 32, 3, 0); // thread 0
        assertInt(a0Start, 0);
        assertInt(a0End, 11);
    }
    {
        SPLIT_RANGE_TO_THREADS(a1Start, a1End, 0, 32, 3, 1); // thread 1
        assertInt(a1Start, 11);
        assertInt(a1End, 22);
    }
    {
        SPLIT_RANGE_TO_THREADS(a2Start, a2End, 0, 32, 3, 2); // thread 2
        assertInt(a2Start, 22);
        assertInt(a2End, 32);
    }

    // <0; 4> across 8 threads
    {
        SPLIT_RANGE_TO_THREADS(b0Start, b0End, 0, 4, 8, 0); // thread 0
        assertInt(b0Start, 0);
        assertInt(b0End, 1);
    }
    {
        SPLIT_RANGE_TO_THREADS(b0Start, b0End, 0, 4, 8, 3); // thread 3
        assertInt(b0Start, 3);
        assertInt(b0End, 4);
    }
    {
        SPLIT_RANGE_TO_THREADS(b0Start, b0End, 0, 4, 8, 4); // thread 4
        assertInt(b0Start, 4); 
        assertInt(b0End, 4);
    }
    {
        SPLIT_RANGE_TO_THREADS(b0Start, b0End, 0, 4, 8, 7); // thread 7
        assertInt(b0Start, 4);
        assertInt(b0End, 4);
    }

    printf("✅ SPLIT_RANGE_TO_THREADS\n");
}

int main() {
    initQuants();

    testRms();
    testMatmulQ80();
    testAdd();
    testSplitRangeToThreads();
    return EXIT_SUCCESS;
}