#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "funcs.hpp"
#include "transformer.hpp"

void readData(float* target, int n, const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) exit(-1);
    fread(target, sizeof(float), n, f);
    fclose(f);
    printf("Read %d floats from %s\n", n, path);
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
    spec.sliceCount = 4;

    SharedBuffer sharedBuffer(SB_LENGTH);
    sharedBuffer.create(SB_XB, 0, 1, spec.dim * sizeof(float));
    sharedBuffer.create(SB_XB2, 0, 1, spec.dim * sizeof(float));
    sharedBuffer.create(SB_HH, 0, 1, spec.hidden_dim * sizeof(float));

    TransformerBlockQkv** qkvs = new TransformerBlockQkv*[spec.sliceCount];
    TransformerBlockAtt** atts = new TransformerBlockAtt*[spec.sliceCount];
    TransformerBlockFfn** ffns = new TransformerBlockFfn*[spec.sliceCount];
    TransformerBlockFfn2** ffn2s = new TransformerBlockFfn2*[spec.sliceCount];
    for (int s = 0; s < spec.sliceCount; s++) {
        qkvs[s] = new TransformerBlockQkv(s, &spec, &sharedBuffer);
        atts[s] = new TransformerBlockAtt(s, &spec, &sharedBuffer);
        ffns[s] = new TransformerBlockFfn(s, &spec, &sharedBuffer);
        ffn2s[s] = new TransformerBlockFfn2(s, &spec, &sharedBuffer);
    }
    TransformerBlockQkv* firstQkv = qkvs[0];

    sharedBuffer.create(SB_Q, 0, spec.sliceCount, firstQkv->qSlice->n * firstQkv->qSlice->d0 * sizeof(float));
    sharedBuffer.create(SB_K, 0, spec.sliceCount, firstQkv->kSlice->n * firstQkv->kSlice->d0 * sizeof(float));
    sharedBuffer.create(SB_V, 0, spec.sliceCount, firstQkv->vSlice->n * firstQkv->vSlice->d0 * sizeof(float));
    sharedBuffer.create(SB_HB, 0, spec.sliceCount, spec.hidden_dim * sizeof(float));

    float x[spec.dim];
    float expectedOutput[spec.dim];

    readData(x, spec.dim, "test-data/block-input.data");
    readData(expectedOutput, spec.dim, "test-data/block-output.data");

    TransformerBlock block(&spec, &sharedBuffer, qkvs, atts, ffns, ffn2s);
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
