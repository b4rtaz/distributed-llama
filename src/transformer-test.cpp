#include "transformer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

void testRopeSlice() {
    TransformerSpec spec;
    spec.dim = 4096;
    spec.headSize = 128;
    spec.nKvHeads = 8;
    spec.seqLen = 2048;
    spec.nHeads = spec.dim / spec.headSize;
    spec.kvDim = (spec.dim * spec.nKvHeads) / spec.nHeads;
    spec.ropeTheta = 10000.0f;

    float* q = new float[spec.dim];
    float* k = new float[spec.kvDim];
    float* correctQ = new float[spec.dim];
    float* correctK = new float[spec.kvDim];
    const int nSliceTests = 5;
    const int nPosTests = 6;
    const int nThreadTests = 3;

    for (int pos = 0; pos < spec.seqLen; pos += spec.seqLen / nPosTests) {
        for (int si = 0; si < nSliceTests; si++) {
            spec.nSlices = pow(2, si);

            for (int nThreads = 1; nThreads <= nThreadTests; nThreads++) {
                printf("pos=%d slices=%d threads=%d\n", pos, spec.nSlices, nThreads);

                for (int j = 0; j < spec.dim; j++) q[j] = (j / (float)spec.dim);
                for (int j = 0; j < spec.kvDim; j++) k[j] = (j / (float)spec.kvDim);

                for (uint8_t sliceIndex = 0; sliceIndex < spec.nSlices; sliceIndex++) {
                    RopeSlice slice(&spec, sliceIndex);
                    for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
                        slice.forward(
                            true,
                            &q[sliceIndex * spec.dim / spec.nSlices],
                            pos, nThreads, threadIndex);
                        slice.forward(
                            false,
                            &k[sliceIndex * spec.kvDim / spec.nSlices],
                            pos, nThreads, threadIndex);
                    }
                }

                if (si == 0 && nThreads == 1) {
                    memcpy(correctQ, q, spec.dim * sizeof(float));
                    memcpy(correctK, k, spec.kvDim * sizeof(float));
                } else {
                    for (int j = 0; j < spec.dim; j++) {
                        if (fabs(q[j] - correctQ[j]) > 1e-6) {
                            printf("q[%d] mismatch: %f != %f\n", j, q[j], correctQ[j]);
                            exit(EXIT_FAILURE);
                        }
                    }
                    for (int j = 0; j < spec.kvDim; j++) {
                        if (fabs(k[j] - correctK[j]) > 1e-6) {
                            printf("k[%d] mismatch: %f != %f\n", j, k[j], correctK[j]);
                            exit(EXIT_FAILURE);
                        }
                    }
                }
            }
        }
    }

    delete[] q;
    delete[] k;
    delete[] correctQ;
    delete[] correctK;
    printf("✅ ropeSlice\n");
}

int main() {
    testRopeSlice();
    return 0;
}
