#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "transformer.hpp"

void readData(float* target, int n, const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) exit(-1);
    fread(target, sizeof(float), n, f);
    fclose(f);
    printf("Read %d floats from %s\n", n, path);
}

long timeMs() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000LL + te.tv_usec / 1000;
}

int main() {
    TransformerSpec spec;
    spec.dim = 4096;
    spec.head_size = 128;
    spec.n_kv_heads = 32;
    spec.seq_len = 2048;
    spec.hidden_dim = 11008;
    spec.n_heads = spec.dim / spec.head_size;
    spec.kv_dim = (spec.dim * spec.n_kv_heads) / spec.n_heads;
    spec.vocab_size = 32000;

    float x[spec.dim];
    float expectedOutput[spec.dim];

    readData(x, spec.dim, "test-data/block-input.data");
    readData(expectedOutput, spec.dim, "test-data/block-output.data");

    TransformerBlock block(&spec, true);
    FILE *fWeights = fopen("test-data/block-weights.data", "r");
    block.readWeights(fWeights);
    fclose(fWeights);

    printf("Weights are read\n");

    long t0 = timeMs();
    block.forward(0, x);
    long t1 = timeMs();

    printf("Forward pass took %ld ms\n", t1 - t0);

    int ix = -1;
    for (int i = 0; i < spec.dim; i++) {
        if (x[i] != expectedOutput[i]) {
            ix = i;
            break;
        }
    }
    if (ix < 0) {
        printf("✅\n");
    } else {
        printf("❌ ix=%d\n", ix);
        printf("%f != %f\n", x[ix], expectedOutput[ix]);
        exit(-1);
    }
}
