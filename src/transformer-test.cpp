#include "transformer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

void testRopeSlice(const TransformerArchType archType, const int nSliceTests, const int nPosTests, const int nThreadTests) {
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

    for (int pos = 0; pos < spec.seqLen; pos += spec.seqLen / nPosTests) {
        for (int si = 0; si < nSliceTests; si++) {
            spec.nSlices = pow(2, si);

            for (int nThreads = 1; nThreads <= nThreadTests; nThreads++) {
                printf("pos=%d nSlices=%d threads=%d\n", pos, spec.nSlices, nThreads);

                for (int j = 0; j < spec.dim; j++) q[j] = 1.0;
                for (int j = 0; j < spec.kvDim; j++) k[j] = 1.0;

                for (uint8_t sliceIndex = 0; sliceIndex < spec.nSlices; sliceIndex++) {
                    RopeSlice* slice;
                    if (archType == LLAMA) {
                        slice = new LlamaRopeSlice(&spec, sliceIndex);
                    } else if (archType == MIXTRAL) {
                        slice = new FalconRopeSlice(&spec, sliceIndex);
                    }

                    for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
                        slice->forward(
                            true,
                            &q[(sliceIndex * spec.dim) / spec.nSlices],
                            pos, nThreads, threadIndex);
                        slice->forward(
                            false,
                            &k[(sliceIndex * spec.kvDim) / spec.nSlices],
                            pos, nThreads, threadIndex);
                    }

                    delete slice;
                }

                if (si == 0 && nThreads == 1) {
                    memcpy(correctQ, q, spec.dim * sizeof(float));
                    memcpy(correctK, k, spec.kvDim * sizeof(float));
                } else {
                    for (int j = 0; j < spec.dim; j++) {
                        if (fabs(q[j] - correctQ[j]) > 1e-6) {
                            printf("q[%d] mismatch: %f != %f (arch=%d)\n", j, q[j], correctQ[j], archType);
                            exit(EXIT_FAILURE);
                        }
                    }
                    for (int j = 0; j < spec.kvDim; j++) {
                        if (fabs(k[j] - correctK[j]) > 1e-6) {
                            printf("k[%d] mismatch: %f != %f (arch=%d)\n", j, k[j], correctK[j], archType);
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
    printf("✅ ropeSlice (arch=%d)\n", archType);
}

int main() {
    testRopeSlice(MIXTRAL, 4, 6, 3);
    testRopeSlice(LLAMA, 6, 4, 3);
    return 0;
}
