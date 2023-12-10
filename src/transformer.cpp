#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include "quants.hpp"
#include "funcs.hpp"
#include "transformer.hpp"

#define NEW_BUFFER(size) newBuffer(size)
#define FREE_BUFFER(buffer) free(buffer)

char* newBuffer(size_t size) {
    char* buffer;
    if (posix_memalign((void**)&buffer, 128, size) != 0) {
        fprintf(stderr, "error: posix_memalign failed\n");
        exit(EXIT_FAILURE);
    }
    if (mlock(buffer, size) != 0) {
        fprintf(stderr, "🚧 Cannot allocate %zu bytes in RAM\n", size);
        // exit(EXIT_FAILURE);
    }
    return buffer;
}

SharedBuffer* createTransformerSharedBuffer(TransformerSpec* spec) {
    SharedBuffer* sb = new SharedBuffer(SB_LENGTH);
    sb->createUnit(SB_UNIT_XB, spec->dim * sizeof(float));
    sb->createUnit(SB_UNIT_HH, spec->hiddenDim * sizeof(float));
    sb->createSliced(SB_SLICED_XB2, spec->dim * sizeof(float), spec->sliceCount);
    sb->createSliced(SB_SLICED_Q, spec->dim * spec->dim * sizeof(float), spec->sliceCount);
    sb->createSliced(SB_SLICED_K, spec->dim * spec->kvDim * sizeof(float), spec->sliceCount);
    sb->createSliced(SB_SLICED_V, spec->dim * spec->kvDim * sizeof(float), spec->sliceCount);
    sb->createSliced(SB_SLICED_HB, spec->hiddenDim * sizeof(float), spec->sliceCount);
    return sb;
}

TransformerBlockFragment::TransformerBlockFragment(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->layerIndex = layerIndex;
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;
}

//
// TransformerBlockQkv
//

TransformerBlockQkv::TransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockFragment(layerIndex, sliceIndex, spec, sharedBuffer) {
    qSlice = new MatMulSlice(spec->blockFloatType, spec->sliceCount, spec->dim, spec->dim);
    kSlice = new MatMulSlice(spec->blockFloatType, spec->sliceCount, spec->dim, spec->kvDim);
    vSlice = new MatMulSlice(spec->blockFloatType, spec->sliceCount, spec->dim, spec->kvDim);
}

TransformerBlockQkv::~TransformerBlockQkv() {
    delete qSlice;
    delete kSlice;
    delete vSlice;
}

NativeTransformerBlockQkv::NativeTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockQkv(layerIndex, sliceIndex, spec, sharedBuffer) {
    qWeights0 = NEW_BUFFER(qSlice->weights0Bytes);
    kWeights0 = NEW_BUFFER(kSlice->weights0Bytes);
    vWeights0 = NEW_BUFFER(vSlice->weights0Bytes);
}

NativeTransformerBlockQkv::~NativeTransformerBlockQkv() {
    FREE_BUFFER(qWeights0);
    FREE_BUFFER(kWeights0);
    FREE_BUFFER(vWeights0);
}

void NativeTransformerBlockQkv::readWeights(char *qWeights, char *kWeights, char *vWeights) {
    this->qSlice->splitWeights(sliceIndex, qWeights, qWeights0);
    this->qSlice->splitWeights(sliceIndex, kWeights, kWeights0);
    this->qSlice->splitWeights(sliceIndex, vWeights, vWeights0);
}

void NativeTransformerBlockQkv::beginForwarding() {
    float *xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float *q0 = (float*)sharedBuffer->getSliced(SB_SLICED_Q, sliceIndex);
    float *k0 = (float*)sharedBuffer->getSliced(SB_SLICED_K, sliceIndex);
    float *v0 = (float*)sharedBuffer->getSliced(SB_SLICED_V, sliceIndex);

    matmul(spec->blockFloatType, q0, xb, qWeights0, qSlice->n, qSlice->d0);
    sharedBuffer->send(SB_SLICED_Q);

    matmul(spec->blockFloatType, k0, xb, kWeights0, kSlice->n, kSlice->d0);
    sharedBuffer->send(SB_SLICED_K);

    matmul(spec->blockFloatType, v0, xb, vWeights0, vSlice->n, vSlice->d0);
    sharedBuffer->send(SB_SLICED_V);
}

void NativeTransformerBlockQkv::waitForEnd() {}

//
// TransformerBlockAtt
//

TransformerBlockAtt::TransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockFragment(layerIndex, sliceIndex, spec, sharedBuffer) {
    woSlice = new MatMulSlice(spec->blockFloatType, spec->sliceCount, spec->dim, spec->dim);
}

TransformerBlockAtt::~TransformerBlockAtt() {
    delete woSlice;
}

NativeTransformerBlockAtt::NativeTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockAtt(layerIndex, sliceIndex, spec, sharedBuffer) {
    woWeights0 = NEW_BUFFER(woSlice->weights0Bytes);
}

NativeTransformerBlockAtt::~NativeTransformerBlockAtt() {
    FREE_BUFFER(woWeights0);
}

void NativeTransformerBlockAtt::readWeights(char* woWeights) {
    this->woSlice->splitWeights(sliceIndex, woWeights, woWeights0);
}

void NativeTransformerBlockAtt::beginForwarding() {
    float *xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float *xb2 = (float*)sharedBuffer->getSliced(SB_SLICED_XB2, sliceIndex);

    matmul(spec->blockFloatType, xb2, xb, woWeights0, woSlice->n, woSlice->d0);
    sharedBuffer->send(SB_SLICED_XB2);
}

void NativeTransformerBlockAtt::waitForEnd() {}

//
// TransformerBlockFfn
//

TransformerBlockFfn::TransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockFragment(layerIndex, sliceIndex, spec, sharedBuffer) {
    w1Slice = new MatMulSlice(spec->blockFloatType, spec->sliceCount, spec->dim, spec->hiddenDim);
    w3Slice = new MatMulSlice(spec->blockFloatType, spec->sliceCount, spec->dim, spec->hiddenDim);
}

TransformerBlockFfn::~TransformerBlockFfn() {
    delete w1Slice;
    delete w3Slice;
}

NativeTransformerBlockFfn::NativeTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockFfn(layerIndex, sliceIndex, spec, sharedBuffer) {
    hb20 = new float[w3Slice->d0];
    w1Weights0 = NEW_BUFFER(w1Slice->weights0Bytes);
    w3Weights0 = NEW_BUFFER(w3Slice->weights0Bytes);
}

NativeTransformerBlockFfn::~NativeTransformerBlockFfn() {
    delete[] hb20;
    FREE_BUFFER(w1Weights0);
    FREE_BUFFER(w3Weights0);
}

void NativeTransformerBlockFfn::readWeights(char *w1Weights, char *w3Weights) {
    this->w1Slice->splitWeights(sliceIndex, w1Weights, w1Weights0);
    this->w3Slice->splitWeights(sliceIndex, w3Weights, w3Weights0);
}

void NativeTransformerBlockFfn::beginForwarding() {
    float* xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float* hb0 = (float*)sharedBuffer->getSliced(SB_SLICED_HB, sliceIndex);

    matmul(spec->blockFloatType, hb0, xb, w1Weights0, w1Slice->n, w1Slice->d0);
    matmul(spec->blockFloatType, hb20, xb, w3Weights0, w3Slice->n, w3Slice->d0);

    // SwiGLU non-linearity
    for (int i = 0; i < w1Slice->d0; i++) {
        float val = hb0[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= hb20[i];
        hb0[i] = val;
    }

    sharedBuffer->send(SB_SLICED_HB);
}

void NativeTransformerBlockFfn::waitForEnd() {}

//
// TransformerBlockFfn2
//

TransformerBlockFfn2::TransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockFragment(layerIndex, sliceIndex, spec, sharedBuffer) {
    w2Slice = new MatMulSlice(spec->blockFloatType, spec->sliceCount, spec->hiddenDim, spec->dim);
}

TransformerBlockFfn2::~TransformerBlockFfn2() {
    delete w2Slice;
}

NativeTransformerBlockFfn2::NativeTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer)
    : TransformerBlockFfn2(layerIndex, sliceIndex, spec, sharedBuffer) {
    w2Weights0 = NEW_BUFFER(w2Slice->weights0Bytes);
}

NativeTransformerBlockFfn2::~NativeTransformerBlockFfn2() {
    FREE_BUFFER(w2Weights0);
}

void NativeTransformerBlockFfn2::readWeights(char *w2Weights) {
    this->w2Slice->splitWeights(sliceIndex, w2Weights, w2Weights0);
}

void NativeTransformerBlockFfn2::beginForwarding() {
    float *hh = (float*)sharedBuffer->getUnit(SB_UNIT_HH);
    float *xb2 = (float*)sharedBuffer->getSliced(SB_SLICED_XB2, sliceIndex);

    matmul(spec->blockFloatType, xb2, hh, w2Weights0, w2Slice->n, w2Slice->d0);

    sharedBuffer->send(SB_SLICED_XB2);
}

void NativeTransformerBlockFfn2::waitForEnd() {}

//
// TransformerBlock
//

TransformerBlock::TransformerBlock(int layerIndex, TransformerSpec* spec, SharedBuffer* sharedBuffer) {
    this->layerIndex = layerIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    rmsAttWeight = new float[spec->dim];
    rmsFfnWeight = new float[spec->dim];

    qkvs = new TransformerBlockQkv*[spec->sliceCount];
    atts = new TransformerBlockAtt*[spec->sliceCount];
    ffns = new TransformerBlockFfn*[spec->sliceCount];
    ffn2s = new TransformerBlockFfn2*[spec->sliceCount];
    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s] = new NativeTransformerBlockQkv(this->layerIndex, s, spec, sharedBuffer);
        atts[s] = new NativeTransformerBlockAtt(this->layerIndex, s, spec, sharedBuffer);
        ffns[s] = new NativeTransformerBlockFfn(this->layerIndex, s, spec, sharedBuffer);
        ffn2s[s] = new NativeTransformerBlockFfn2(this->layerIndex, s, spec, sharedBuffer);
    }

    xb2 = new float[spec->dim];
    hb = new float[spec->hiddenDim];
    q = new float[spec->dim];
    keyCache = new float[spec->seqLen * spec->kvDim];
    valueCache = new float[spec->seqLen * spec->kvDim];
    att = new float[spec->nHeads * spec->seqLen];
}

TransformerBlock::~TransformerBlock() {
    for (int s = 0; s < spec->sliceCount; s++) {
        delete qkvs[s];
        delete atts[s];
        delete ffns[s];
        delete ffn2s[s];
    }
    delete[] qkvs;
    delete[] atts;
    delete[] ffns;
    delete[] ffn2s;

    delete[] rmsAttWeight;
    delete[] rmsFfnWeight;

    delete[] xb2;
    delete[] hb;
    delete[] q;
    delete[] keyCache;
    delete[] valueCache;
    delete[] att;
}

long TransformerBlock::readWeights(char* wd) {
    long fs = getFloatBytes(spec->blockFloatType);

    char* w = wd;
    memcpy(rmsAttWeight, w, spec->dim * sizeof(float));
    w += spec->dim * sizeof(float);

    memcpy(rmsFfnWeight, w, spec->dim * sizeof(float));
    w += spec->dim * sizeof(float);

    char* wq = w;
    w += spec->dim * spec->dim * fs;
    char* wk = w;
    w += spec->dim * spec->nKvHeads * spec->headSize * fs;
    char* wv = w;
    w += spec->dim * spec->nKvHeads * spec->headSize * fs;

    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s]->readWeights(wq, wk, wv);
    }

    char* wo = w;
    w += spec->dim * spec->dim * fs;

    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->readWeights(wo);
    }

    char* w1 = w;
    w += spec->dim * spec->hiddenDim * fs;
    char* w2 = w;
    w += spec->hiddenDim * spec->dim * fs;
    char* w3 = w;
    w += spec->dim * spec->hiddenDim * fs;

    for (int s = 0; s < spec->sliceCount; s++) {
        ffns[s]->readWeights(w1, w3);
        ffn2s[s]->readWeights(w2);
    }

    return (long)(w - wd);
}

void TransformerBlock::forward(int pos, float* x) {
    int dim = spec->dim;
    int kvDim = spec->kvDim;
    int kvMul = spec->nHeads / spec->nKvHeads; // integer multiplier of the kv sharing in multiquery
    int hiddenDim =  spec->hiddenDim;
    int headSize = dim / spec->nHeads;

    float* xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float* hh = (float*)sharedBuffer->getUnit(SB_UNIT_HH);

    // attention rmsnorm
    rmsnorm(xb, x, rmsAttWeight, dim);
    sharedBuffer->send(SB_UNIT_XB);

    // qkv matmuls for this position
    float* k = keyCache + pos * kvDim;
    float* v = valueCache + pos * kvDim;

    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s]->beginForwarding();
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        TransformerBlockQkv *qkv = qkvs[s];
        qkv->waitForEnd();
        qkv->qSlice->mergeOutputs(s, q, (float*)sharedBuffer->getSliced(SB_SLICED_Q, s));
        qkv->kSlice->mergeOutputs(s, k, (float*)sharedBuffer->getSliced(SB_SLICED_K, s));
        qkv->vSlice->mergeOutputs(s, v, (float*)sharedBuffer->getSliced(SB_SLICED_V, s));
    }

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % headSize;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)headSize);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int _v = 0; _v < rotn; _v++) {
            float* vec = _v == 0 ? q : k; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }

    // multihead attention. iterate over all heads
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < spec->nHeads; h++) {
        // get the query vector for this head
        float* _q = q + h * headSize;
        // attention scores for this head
        float* _att = att + h * spec->seqLen;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = keyCache + t * kvDim + (h / kvMul) * headSize;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < headSize; i++) {
                score += _q[i] * k[i];
            }
            score /= sqrtf(headSize);
            // save the score to the attention buffer
            _att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(_att, pos + 1);

        // weighted sum of the values, store back into xb
        float* _xb = xb + h * headSize;
        memset(_xb, 0, headSize * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* _v = valueCache + t * kvDim + (h / kvMul) * headSize;
            // get the attention weight for this timestep
            float a = _att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < headSize; i++) {
                _xb[i] += a * _v[i];
            }
        }
    }

    sharedBuffer->send(SB_UNIT_XB);

    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->beginForwarding();
    }

    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->waitForEnd();
        atts[s]->woSlice->mergeOutputs(s, xb2, (float*)sharedBuffer->getSliced(SB_SLICED_XB2, s));
    }

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
        x[i] += xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(xb, x, rmsFfnWeight, dim);

    sharedBuffer->send(SB_UNIT_XB);

    for (int s = 0; s < spec->sliceCount; s++) {
        ffns[s]->beginForwarding();
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        ffns[s]->waitForEnd();
        ffns[s]->w3Slice->mergeOutputs(s, hb, (float*)sharedBuffer->getSliced(SB_SLICED_HB, s));
    }

    memcpy(hh, hb, hiddenDim * sizeof(float));
    sharedBuffer->send(SB_UNIT_HH);

    for (int s = 0; s < spec->sliceCount; s++) {
        ffn2s[s]->beginForwarding();
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        ffn2s[s]->waitForEnd();
        ffn2s[s]->w2Slice->mergeOutputs(s, xb, (float*)sharedBuffer->getSliced(SB_SLICED_XB2, s));
    }

    // residual connection
    for (int i = 0; i < dim; i++) {
        x[i] += xb[i];
    }
}

//
// Transformer
//

Transformer::Transformer(TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    this->blocks = new TransformerBlock*[spec->nLayers];
    for (int l = 0; l < spec->nLayers; l++) {
        this->blocks[l] = new TransformerBlock(l, spec, sharedBuffer);
    }

    this->x = new float[spec->dim];
    this->token_embedding_table = new float[spec->vocabSize * spec->dim];
    this->rms_final_weight = new float[spec->dim];
    this->wcls = new float[spec->vocabSize * spec->dim];

    this->logits = new float[spec->vocabSize];
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
    delete[] logits;
}

long Transformer::readWeights(char* wd, bool sharedWeights) {
    char *w = wd;

    memcpy(token_embedding_table, w, spec->vocabSize * spec->dim * sizeof(float));
    w += spec->vocabSize * spec->dim * sizeof(float);

    for (int i = 0; i < spec->nLayers; i++) {
        w += blocks[i]->readWeights(w);
    }

    memcpy(rms_final_weight, w, spec->dim * sizeof(float));
    w += spec->dim * sizeof(float);

    w += (spec->seqLen * spec->headSize / 2) * sizeof(float); // skip what used to be freq_cis_real (for RoPE)
    w += (spec->seqLen * spec->headSize / 2) * sizeof(float); // skip what used to be freq_cis_imag (for RoPE)

    memcpy(wcls, sharedWeights ? (char*)token_embedding_table : w, spec->vocabSize * spec->dim * sizeof(float));
    if (!sharedWeights) {
        w += spec->vocabSize * spec->dim * sizeof(float);
    }

    return (long)(w - wd);
}

void Transformer::forward(int token, int pos) {
    float* content_row = token_embedding_table + token * spec->dim;
    memcpy(x, content_row, spec->dim * sizeof(float));

    for (int i = 0; i < spec->nLayers; i++) {
        blocks[i]->forward(pos, x);
    }

    rmsnorm(x, x, rms_final_weight, spec->dim);

    // classifier into logits
    matmul(F32, logits, x, (char*)wcls, spec->dim, spec->vocabSize);
}
