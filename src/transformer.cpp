#include <math.h>
#include <string.h>
#include "funcs.hpp"
#include "transformer.hpp"

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

TransformerBlockQkv::TransformerBlockQkv(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    qSlice = new MatMulSlice(spec->sliceCount, spec->dim, spec->dim);
    kSlice = new MatMulSlice(spec->sliceCount, spec->dim, spec->kvDim);
    vSlice = new MatMulSlice(spec->sliceCount, spec->dim, spec->kvDim);

    qWeights0 = new float[qSlice->weights0Length];
    kWeights0 = new float[kSlice->weights0Length];
    vWeights0 = new float[vSlice->weights0Length];
}

TransformerBlockQkv::~TransformerBlockQkv() {
    delete qSlice;
    delete kSlice;
    delete vSlice;
    delete[] qWeights0;
    delete[] kWeights0;
    delete[] vWeights0;
}

void TransformerBlockQkv::readWeights(float *qWeights, float *kWeights, float *vWeights) {
    this->qSlice->splitWeights(sliceIndex, qWeights, qWeights0);
    this->qSlice->splitWeights(sliceIndex, kWeights, kWeights0);
    this->qSlice->splitWeights(sliceIndex, vWeights, vWeights0);
}

void TransformerBlockQkv::beginForwarding() {
    float *xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float *q0 = (float*)sharedBuffer->getSliced(SB_SLICED_Q, sliceIndex);
    float *k0 = (float*)sharedBuffer->getSliced(SB_SLICED_K, sliceIndex);
    float *v0 = (float*)sharedBuffer->getSliced(SB_SLICED_V, sliceIndex);

    matmul(q0, xb, qWeights0, qSlice->n, qSlice->d0);
    sharedBuffer->send(SB_SLICED_Q);

    matmul(k0, xb, kWeights0, kSlice->n, kSlice->d0);
    sharedBuffer->send(SB_SLICED_K);

    matmul(v0, xb, vWeights0, vSlice->n, vSlice->d0);
    sharedBuffer->send(SB_SLICED_V);
}

void TransformerBlockQkv::waitForEnd() {
    // TODO
}

TransformerBlockAtt::TransformerBlockAtt(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    woSlice = new MatMulSlice(spec->sliceCount, spec->dim, spec->dim);

    woWeights0 = new float[woSlice->weights0Length];
}

TransformerBlockAtt::~TransformerBlockAtt() {
    delete woSlice;
    delete[] woWeights0;
}

void TransformerBlockAtt::readWeights(float *woWeights) {
    this->woSlice->splitWeights(sliceIndex, woWeights, woWeights0);
}

void TransformerBlockAtt::beginForwarding() {
    float *xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float *xb2 = (float*)sharedBuffer->getSliced(SB_SLICED_XB2, sliceIndex);

    matmul(xb2, xb, woWeights0, woSlice->n, woSlice->d0);
    sharedBuffer->send(SB_SLICED_XB2);
}

void TransformerBlockAtt::waitForEnd() {
    // TODO
}

TransformerBlockFfn::TransformerBlockFfn(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    w1Slice = new MatMulSlice(spec->sliceCount, spec->dim, spec->hiddenDim);
    w3Slice = new MatMulSlice(spec->sliceCount, spec->dim, spec->hiddenDim);

    hb20 = new float[w3Slice->d0];

    w1Weights0 = new float[w1Slice->weights0Length];
    w3Weights0 = new float[w3Slice->weights0Length];
}

TransformerBlockFfn::~TransformerBlockFfn() {
    delete w1Slice;
    delete w3Slice;
    delete[] w1Weights0;
    delete[] w3Weights0;
}

void TransformerBlockFfn::readWeights(float *w1Weights, float *w3Weights) {
    this->w1Slice->splitWeights(sliceIndex, w1Weights, w1Weights0);
    this->w3Slice->splitWeights(sliceIndex, w3Weights, w3Weights0);
}

void TransformerBlockFfn::beginForwarding() {
    float* xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float* hb0 = (float*)sharedBuffer->getSliced(SB_SLICED_HB, sliceIndex);

    matmul(hb0, xb, w1Weights0, w1Slice->n, w1Slice->d0);
    matmul(hb20, xb, w3Weights0, w3Slice->n, w3Slice->d0);

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

void TransformerBlockFfn::waitForEnd() {
    // TODO
}

TransformerBlockFfn2::TransformerBlockFfn2(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    w2Slice = new MatMulSlice(spec->sliceCount, spec->hiddenDim, spec->dim);
    w2Weights0 = new float[w2Slice->weights0Length];
}

TransformerBlockFfn2::~TransformerBlockFfn2() {
    delete w2Slice;
    delete[] w2Weights0;
}

void TransformerBlockFfn2::readWeights(float *w2Weights) {
    this->w2Slice->splitWeights(sliceIndex, w2Weights, w2Weights0);
}

void TransformerBlockFfn2::beginForwarding() {
    float *hh = (float*)sharedBuffer->getUnit(SB_UNIT_HH);
    float *xb2 = (float*)sharedBuffer->getSliced(SB_SLICED_XB2, sliceIndex);

    matmul(xb2, hh, w2Weights0, w2Slice->n, w2Slice->d0);

    sharedBuffer->send(SB_SLICED_XB2);
}

void TransformerBlockFfn2::waitForEnd() {
    // TODO
}

TransformerBlock::TransformerBlock(TransformerSpec* spec, SharedBuffer* sharedBuffer) {
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    rmsAttWeight = new float[spec->dim];
    rmsFfnWeight = new float[spec->dim];

    qkvs = new TransformerBlockQkv*[spec->sliceCount];
    atts = new TransformerBlockAtt*[spec->sliceCount];
    ffns = new TransformerBlockFfn*[spec->sliceCount];
    ffn2s = new TransformerBlockFfn2*[spec->sliceCount];
    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s] = new TransformerBlockQkv(s, spec, sharedBuffer);
        atts[s] = new TransformerBlockAtt(s, spec, sharedBuffer);
        ffns[s] = new TransformerBlockFfn(s, spec, sharedBuffer);
        ffn2s[s] = new TransformerBlockFfn2(s, spec, sharedBuffer);
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
    float* w = (float*)wd;
    memcpy(rmsAttWeight, w, spec->dim * sizeof(float));
    w += spec->dim;

    memcpy(rmsFfnWeight, w, spec->dim * sizeof(float));
    w += spec->dim;

    float* wq = new float[spec->dim * spec->dim];
    float* wk = new float[spec->dim * spec->nKvHeads * spec->headSize];
    float* wv = new float[spec->dim * spec->nKvHeads * spec->headSize];

    memcpy(wq, w, spec->dim * spec->dim * sizeof(float));
    w += spec->dim * spec->dim;

    memcpy(wk, w, spec->dim * spec->nKvHeads * spec->headSize * sizeof(float));
    w += spec->dim * spec->nKvHeads * spec->headSize;

    memcpy(wv, w, spec->dim * spec->nKvHeads * spec->headSize * sizeof(float));
    w += spec->dim * spec->nKvHeads * spec->headSize;

    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s]->readWeights(wq, wk, wv);
    }
    delete[] wq;
    delete[] wk;
    delete[] wv;

    float* wo = new float[spec->dim * spec->dim];
    memcpy(wo, w, spec->dim * spec->dim * sizeof(float));
    w += spec->dim * spec->dim;

    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->readWeights(wo);
    }
    delete[] wo;

    float* w1 = new float[spec->dim * spec->hiddenDim];
    float* w2 = new float[spec->hiddenDim * spec->dim];
    float* w3 = new float[spec->dim * spec->hiddenDim];

    memcpy(w1, w, spec->dim * spec->hiddenDim * sizeof(float));
    w += spec->dim * spec->hiddenDim;

    memcpy(w2, w, spec->hiddenDim * spec->dim * sizeof(float));
    w += spec->hiddenDim * spec->dim;

    memcpy(w3, w, spec->dim * spec->hiddenDim * sizeof(float));
    w += spec->dim * spec->hiddenDim;

    for (int s = 0; s < spec->sliceCount; s++) {
        ffns[s]->readWeights(w1, w3);
        ffn2s[s]->readWeights(w2);
    }
    delete[] w1;
    delete[] w2;
    delete[] w3;
    return (long)((char*)w - wd);
}

void TransformerBlock::forward(int pos, float* x) {
    int dim = spec->dim;
    int kv_dim = spec->kvDim;
    int kv_mul = spec->nHeads / spec->nKvHeads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  spec->hiddenDim;
    int head_size = dim / spec->nHeads;

    float* xb = (float*)sharedBuffer->getUnit(SB_UNIT_XB);
    float* hh = (float*)sharedBuffer->getUnit(SB_UNIT_HH);

    // attention rmsnorm
    rmsnorm(xb, x, rmsAttWeight, dim);
    sharedBuffer->send(SB_UNIT_XB);

    // qkv matmuls for this position
    float* k = keyCache + pos * kv_dim;
    float* v = valueCache + pos * kv_dim;

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
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
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
        float* _q = q + h * head_size;
        // attention scores for this head
        float* _att = att + h * spec->seqLen;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = keyCache + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += _q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            _att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(_att, pos + 1);

        // weighted sum of the values, store back into xb
        float* _xb = xb + h * head_size;
        memset(_xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* _v = valueCache + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = _att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
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

    memcpy(hh, hb, hidden_dim * sizeof(float));
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

Transformer::Transformer(TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    this->blocks = new TransformerBlock*[spec->nLayers];
    for (int i = 0; i < spec->nLayers; i++) {
        this->blocks[i] = new TransformerBlock(spec, sharedBuffer);
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
    matmul(logits, x, wcls, spec->dim, spec->vocabSize);
}
