#include "funcs.hpp"
#include "utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

int main() {
    initQuants();

    testRms();
    testMatmulQ80();
    return EXIT_SUCCESS;
}