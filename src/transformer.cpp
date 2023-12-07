#include <string.h>
#include "funcs.hpp"
#include "transformer.hpp"
#include "transformer-block.hpp"

Transformer::Transformer(
    TransformerSpec* spec,
    SharedBuffer *sharedBuffer) {
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    this->blocks = new TransformerBlock*[spec->nLayers];
    for (int i = 0; i < spec->nLayers; i++) {
        this->blocks[i] = new TransformerBlock(spec, sharedBuffer);
    }

    this->x = new float[spec->dim];
    this->token_embedding_table = new float[spec->vocabSize * spec->dim];
    this->rms_final_weight = new float[spec->dim];
    this->wcls = new float[spec->vocabSize];
}

Transformer::~Transformer() {
    for (int i = 0; i < spec->nLayers; i++) {
        delete blocks[i];
    }
    delete[] blocks;

    delete[] x;
    delete[] token_embedding_table;
    delete[] rms_final_weight;
    delete[] wcls;
}

void Transformer::readWeights(char* wd) {
    char *w = wd;

    memcpy(token_embedding_table, w, spec->vocabSize * spec->dim * sizeof(float));
    w += spec->vocabSize * spec->dim * sizeof(float);

    char* w = wd;
    for (int i = 0; i < spec->nLayers; i++) {
        w += blocks[i]->readWeights(w);
    }

    memcpy(rms_final_weight, w, spec->dim * sizeof(float));
    w += spec->dim * sizeof(float);

    memcpy(wcls, w, spec->vocabSize * sizeof(float));
    w += spec->vocabSize * sizeof(float);
}

void Transformer::forward(int token, int pos) {
    float* content_row = token_embedding_table + token * spec->dim;
    memcpy(x, content_row, spec->dim * sizeof(float));

    for (int i = 0; i < spec->nLayers; i++) {
        blocks[i]->forward(pos, x);
    }

    rmsnorm(x, x, rms_final_weight, spec->dim);

    // classifier into logits
    matmul(logits, x, wcls, spec->dim, spec->vocabSize);
}
