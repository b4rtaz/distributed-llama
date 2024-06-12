#include "commands.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

void testRopeSlice(int arch, const int nSliceTests, const int nPosTests, const int nThreadTests) {
    int dim = 4096;
    int headSize = 128;
    int nKvHeads = 8;
    int seqLen = 2048;
    int nHeads = dim / headSize;
    int kvDim = (dim * nKvHeads) / nHeads;
    int ropeTheta = 10000.0f;

    float* q = new float[dim];
    float* k = new float[kvDim];
    float* correctQ = new float[dim];
    float* correctK = new float[kvDim];

    for (int pos = 0; pos < seqLen; pos += seqLen / nPosTests) {
        for (int si = 0; si < nSliceTests; si++) {
            int nSlices = pow(2, si);

            for (int nThreads = 1; nThreads <= nThreadTests; nThreads++) {
                printf("pos=%d nSlices=%d threads=%d\n", pos, nSlices, nThreads);

                for (int j = 0; j < dim; j++) q[j] = 1.0;
                for (int j = 0; j < kvDim; j++) k[j] = 1.0;

                for (slice_index_t sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
                    RopeSlice slice(dim, kvDim, nKvHeads, nSlices, seqLen, headSize, ropeTheta, sliceIndex);
                    RopeCommand* rope;
                    if (arch == 1) {
                        rope = new LlamaRopeCommand(&slice);
                    } else if (arch == 2) {
                        rope = new FalconRopeCommand(&slice);
                    }

                    for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
                        rope->forward(
                            true,
                            &q[(sliceIndex * dim) / nSlices],
                            pos, nThreads, threadIndex);
                        rope->forward(
                            false,
                            &k[(sliceIndex * kvDim) / nSlices],
                            pos, nThreads, threadIndex);
                    }

                    delete rope;
                }

                if (si == 0 && nThreads == 1) {
                    memcpy(correctQ, q, dim * sizeof(float));
                    memcpy(correctK, k, kvDim * sizeof(float));
                } else {
                    for (int j = 0; j < dim; j++) {
                        if (fabs(q[j] - correctQ[j]) > 1e-6) {
                            printf("q[%d] mismatch: %f != %f (arch=%d)\n", j, q[j], correctQ[j], arch);
                            exit(EXIT_FAILURE);
                        }
                    }
                    for (int j = 0; j < kvDim; j++) {
                        if (fabs(k[j] - correctK[j]) > 1e-6) {
                            printf("k[%d] mismatch: %f != %f (arch=%d)\n", j, k[j], correctK[j], arch);
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
    printf("✅ ropeSlice (arch=%d)\n", arch);
}

int main() {
    testRopeSlice(2, 4, 6, 3);
    testRopeSlice(1, 6, 4, 3);
    return 0;
}
