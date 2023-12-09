#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "funcs.hpp"
#include "shared-buffer.hpp"
#include "transformer.hpp"

void readData(float* target, int n, const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) exit(-1);
    fread(target, sizeof(float), n, f);
    fclose(f);
    printf("Read %lu bytes from %s\n", n * sizeof(float), path);
}

int main() {
    TransformerSpec spec;
    spec.dim = 4096;
    spec.nLayers = 1;
    spec.headSize = 128;
    spec.nKvHeads = 32;
    spec.seqLen = 2048;
    spec.hiddenDim = 11008;
    spec.nHeads = spec.dim / spec.headSize;
    spec.kvDim = (spec.dim * spec.nKvHeads) / spec.nHeads;
    spec.vocabSize = 32000;
    spec.sliceCount = 4;

    SharedBuffer* sharedBuffer = createTransformerSharedBuffer(&spec);
    TransformerBlock block(&spec, sharedBuffer);

    {
        const char* path = "test-data/block-weights.data";
        FILE *fw = fopen(path, "rb");
        fseek(fw, 0, SEEK_END);
        size_t weightsSize = ftell(fw);
        fclose(fw);

        int fWeights = open(path, O_RDONLY);
        if (fWeights == -1) {
            printf("open failed!\n");
            exit(EXIT_FAILURE);
        }

        char* weights = (char*)mmap(NULL, weightsSize, PROT_READ, MAP_PRIVATE, fWeights, 0);
        if (weights == MAP_FAILED) {
            printf("mmap failed!\n");
            exit(EXIT_FAILURE);
        }

        int bytes = block.readWeights(weights);
        printf("Loaded weights (%d bytes)\n", bytes);

        munmap(weights, weightsSize);
    }

    float x[spec.dim];
    float expectedOutput[spec.dim];

    readData(x, spec.dim, "test-data/block-input.data");
    readData(expectedOutput, spec.dim, "test-data/block-output.data");

    long t0 = timeMs();
    block.forward(0, x);
    long t1 = timeMs();

    printf("Forward pass took %ld ms\n", t1 - t0);

    delete sharedBuffer;

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
