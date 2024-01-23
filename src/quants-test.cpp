#include <cstdlib>
#include <cstdio>
#include <cmath>
#include "utils.hpp"
#include "quants.hpp"

void testQ80(const int len, int nThreads) {
    unsigned long long state = 800000010L;
    float input[len];
    float* output = new float[len];
    BlockQ80* q80s = new BlockQ80[len / QK80];

    for (int i = 0; i < len; i++) {
        input[i] = randomF32(&state);
        output[i] = 0;
    }

    for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
        quantizeQ80Row((float*)&input, (BlockQ80*)q80s, len, nThreads, threadIndex);
    }
    for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
        dequantizeQ80Row((BlockQ80*)q80s, (float*)output, len, nThreads, threadIndex);
    }

    for (int i = 0; i < len; i++) {
        float diff = fabs(output[i] - input[i]);
        if (diff > 0.0043) {
            printf("❌ (%d, %d) ix=%d %f != %f diff=%f nThreads=%d\n", len, nThreads, i, output[i], input[i], diff, nThreads);
            exit(EXIT_FAILURE);
        }
    }

    delete[] output;
    delete[] q80s;
}

int main() {
    initQuants();

    testQ80(1024, 1);
    testQ80(1024, 2);
    testQ80(1024, 4);
    testQ80(1024, 8);
    testQ80(1024, 16);
    testQ80(768, 1);
    testQ80(768, 2);
    testQ80(768, 4);
    testQ80(768, 8);
    testQ80(2752, 1);
    testQ80(2752, 2);
    testQ80(2752, 4);

    printf("✅ Q80 quantized correctly\n");
    return EXIT_SUCCESS;
}
