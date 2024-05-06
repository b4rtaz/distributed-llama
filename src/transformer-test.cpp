#include "transformer.hpp"
#include <cmath>

void testRopeSlice() {
    TransformerSpec spec;
    spec.dim = 4096;
    spec.headSize = 128;
    spec.nKvHeads = 32;
    spec.seqLen = 2048;
    spec.nHeads = spec.dim / spec.headSize;
    spec.kvDim = (spec.dim * spec.nKvHeads) / spec.nHeads;

    float* q = new float[spec.dim];
    float* k = new float[spec.kvDim];
    float* q1 = new float[spec.dim];
    float* k1 = new float[spec.kvDim];
    const int nPosTests = 16;
    const int nThreadTests = 4;

    for (int pos = 0; pos < spec.seqLen; pos += spec.seqLen / nPosTests) {
        for (int i = 0; i < 8; i++) {
            spec.nSlices = pow(2, i);
            for (int nThreads = 1; nThreads < nThreadTests; nThreads++) {
                for (int j = 0; j < spec.dim; j++) q[j] = (j / (float)spec.dim);
                for (int j = 0; j < spec.kvDim; j++) k[j] = (j / (float)spec.kvDim);

                for (int s = 0; s < spec.nSlices; s++) {
                    RopeSlice slice(&spec, s);
                    for (int threadIndex = 0; threadIndex < nThreads; threadIndex++) {
                        slice.forward(
                            &q[s * spec.dim / spec.nSlices],
                            &k[s * spec.kvDim / spec.nSlices],
                            pos, nThreads, threadIndex);
                    }
                }

                if (i == 0 && nThreads == 1) {
                    memcpy(q1, q, spec.dim * sizeof(float));
                    memcpy(k1, k, spec.kvDim * sizeof(float));
                } else {
                    for (int j = 0; j < spec.dim; j++) {
                        if (fabs(q[j] - q1[j]) > 1e-6) {
                            printf("q[%d] mismatch: %f != %f\n", j, q[j], q1[j]);
                            exit(EXIT_FAILURE);
                        }
                    }
                    for (int j = 0; j < spec.kvDim; j++) {
                        if (fabs(k[j] - k1[j]) > 1e-6) {
                            printf("k[%d] mismatch: %f != %f\n", j, k[j], k1[j]);
                            exit(EXIT_FAILURE);
                        }
                    }
                }
            }
        }
    }

    delete[] q;
    delete[] k;
    delete[] q1;
    delete[] k1;
    printf("✅ RopeSlice\n");
}

int main() {
    testRopeSlice();
    return 0;
}
