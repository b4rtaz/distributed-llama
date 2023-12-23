#include <cmath>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <cassert>
#include "quants.hpp"
#include "funcs.hpp"
#include "transformer.hpp"

#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

#define NEW_BUFFER(size) newBuffer(size)
#define FREE_BUFFER(buffer) free(buffer)

char* newBuffer(size_t size) {
    char* buffer;
    if (posix_memalign((void**)&buffer, 16, size) != 0) {
        fprintf(stderr, "error: posix_memalign failed\n");
        exit(EXIT_FAILURE);
    }
    if (mlock(buffer, size) != 0) {
        fprintf(stderr, "🚧 Cannot allocate %zu bytes directly in RAM\n", size);
        // exit(EXIT_FAILURE);
    }
    return buffer;
}

NativeTransformerState::NativeTransformerState(SharedBuffer* buffer) {
    this->buffer = buffer;
}

char* NativeTransformerState::getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    return buffer->getSliced(bufferIndex, sliceIndex);
}

char* NativeTransformerState::getUnitBuffer(uint8_t bufferIndex) {
    return buffer->getUnit(bufferIndex);
}

void NativeTransformerState::readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {}
void NativeTransformerState::sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {}
void NativeTransformerState::sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {}

RemoteTransformerState::RemoteTransformerState(SharedBuffer* buffer, RemoteClient* client) {
    this->buffer = buffer;
    this->client = client;
}

RemoteTransformerState::~RemoteTransformerState() {
    delete buffer;
}

char* RemoteTransformerState::getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex)  {
    return buffer->getSliced(bufferIndex, sliceIndex);
}

char* RemoteTransformerState::getUnitBuffer(uint8_t bufferIndex)  {
    return buffer->getUnit(bufferIndex);
}

void RemoteTransformerState::readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    char* data = buffer->getSliced(bufferIndex, sliceIndex);
    size_t bytes = buffer->getBytes(bufferIndex) / buffer->getSlices(bufferIndex);
    client->readBuffer(sliceIndex, bufferIndex, data, bytes);
}

void RemoteTransformerState::sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    size_t bytes = buffer->getBytes(bufferIndex);
    char* data = buffer->getUnit(bufferIndex);
    client->sendBuffer(sliceIndex, bufferIndex, data, bytes);
}

void RemoteTransformerState::sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    printf("Invalid state: sendSlicedBuffer\n");
    exit(EXIT_FAILURE);
} 

SharedBuffer* initSharedBuffer(TransformerSpec* spec) {
    SharedBuffer* buffer = new SharedBuffer(SB_LENGTH);
    buffer->createUnit(SB_UNIT_XB, spec->dim * sizeof(float));
    buffer->createUnit(SB_UNIT_HH, spec->hiddenDim * sizeof(float));
    buffer->createSliced(SB_SLICED_XB2, spec->dim * sizeof(float), spec->sliceCount);
    buffer->createSliced(SB_SLICED_Q, spec->dim * sizeof(float), spec->sliceCount);
    buffer->createSliced(SB_SLICED_K, spec->kvDim * sizeof(float), spec->sliceCount);
    buffer->createSliced(SB_SLICED_V, spec->kvDim * sizeof(float), spec->sliceCount);
    buffer->createSliced(SB_SLICED_HB, spec->hiddenDim * sizeof(float), spec->sliceCount);
    return buffer;
}

//
// TransformerFragment
//

TransformerFragment::TransformerFragment(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState *state) {
    this->layerIndex = layerIndex;
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->state = state;
}

//
// TransformerBlockQkv
//

TransformerBlockQkv::TransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState *state)
    : TransformerFragment(layerIndex, sliceIndex, spec, state) {
    qSlice = new MatMulSlice(spec->floatType, spec->sliceCount, spec->dim, spec->dim);
    kSlice = new MatMulSlice(spec->floatType, spec->sliceCount, spec->dim, spec->kvDim);
    vSlice = new MatMulSlice(spec->floatType, spec->sliceCount, spec->dim, spec->kvDim);
}

TransformerBlockQkv::~TransformerBlockQkv() {
    delete qSlice;
    delete kSlice;
    delete vSlice;
}

NativeTransformerBlockQkv::NativeTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState *state)
    : TransformerBlockQkv(layerIndex, sliceIndex, spec, state) {
    this->config = config;
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
    this->kSlice->splitWeights(sliceIndex, kWeights, kWeights0);
    this->vSlice->splitWeights(sliceIndex, vWeights, vWeights0);
}

void NativeTransformerBlockQkv::beginForwarding() {
    float *xb = (float*)state->getUnitBuffer(SB_UNIT_XB);
    float *q0 = (float*)state->getSlicedBuffer(SB_SLICED_Q, sliceIndex);
    float *k0 = (float*)state->getSlicedBuffer(SB_SLICED_K, sliceIndex);
    float *v0 = (float*)state->getSlicedBuffer(SB_SLICED_V, sliceIndex);

    long t0 = timeMs();
    matmul(spec->floatType, config->nThread, q0, xb, qWeights0, qSlice->n, qSlice->d0);
    matmul(spec->floatType, config->nThread, k0, xb, kWeights0, kSlice->n, kSlice->d0);
    matmul(spec->floatType, config->nThread, v0, xb, vWeights0, vSlice->n, vSlice->d0);
    long t1 = timeMs();
    printf("qkv %d ms\n", (int)(t1 - t0));

    state->sendSlicedBuffer(SB_SLICED_Q, sliceIndex);
    state->sendSlicedBuffer(SB_SLICED_K, sliceIndex);
    state->sendSlicedBuffer(SB_SLICED_V, sliceIndex);
}

void NativeTransformerBlockQkv::waitForEnd() {}

RemoteTransformerBlockQkv::RemoteTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client)
    : TransformerBlockQkv(layerIndex, sliceIndex, spec, state) {
    this->client = client;
}

void RemoteTransformerBlockQkv::readWeights(char* qWeights, char* kWeights, char* vWeights) {
    size_t weightsBytes = this->qSlice->weights0Bytes + this->kSlice->weights0Bytes + this->vSlice->weights0Bytes;
    char* weights = new char[weightsBytes];

    this->qSlice->splitWeights(sliceIndex, qWeights, &weights[0]);
    this->kSlice->splitWeights(sliceIndex, kWeights, &weights[this->qSlice->weights0Bytes]);
    this->vSlice->splitWeights(sliceIndex, vWeights, &weights[this->qSlice->weights0Bytes + this->kSlice->weights0Bytes]);

    client->createFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_QKV, weights, weightsBytes);
    delete[] weights;
}

void RemoteTransformerBlockQkv::beginForwarding() {
    state->sendUnitBuffer(SB_UNIT_XB, sliceIndex);
    client->forwardFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_QKV);
}

void RemoteTransformerBlockQkv::waitForEnd() {
    state->readSlicedBuffer(SB_SLICED_Q, sliceIndex);
    state->readSlicedBuffer(SB_SLICED_K, sliceIndex);
    state->readSlicedBuffer(SB_SLICED_V, sliceIndex);
}

//
// TransformerBlockAtt
//

TransformerBlockAtt::TransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState *state)
    : TransformerFragment(layerIndex, sliceIndex, spec, state) {
    woSlice = new MatMulSlice(spec->floatType, spec->sliceCount, spec->dim, spec->dim);
}

TransformerBlockAtt::~TransformerBlockAtt() {
    delete woSlice;
}

NativeTransformerBlockAtt::NativeTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState *state)
    : TransformerBlockAtt(layerIndex, sliceIndex, spec, state) {
    this->config = config;
    woWeights0 = NEW_BUFFER(woSlice->weights0Bytes);
}

NativeTransformerBlockAtt::~NativeTransformerBlockAtt() {
    FREE_BUFFER(woWeights0);
}

void NativeTransformerBlockAtt::readWeights(char* woWeights) {
    this->woSlice->splitWeights(sliceIndex, woWeights, woWeights0);
}

void NativeTransformerBlockAtt::beginForwarding() {
    float *xb = (float*)state->getUnitBuffer(SB_UNIT_XB);
    float *xb2 = (float*)state->getSlicedBuffer(SB_SLICED_XB2, sliceIndex);

    matmul(spec->floatType, config->nThread, xb2, xb, woWeights0, woSlice->n, woSlice->d0);
    state->sendSlicedBuffer(SB_SLICED_XB2, sliceIndex);
}

void NativeTransformerBlockAtt::waitForEnd() {}

RemoteTransformerBlockAtt::RemoteTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client)
    : TransformerBlockAtt(layerIndex, sliceIndex, spec, state) {
    this->client = client;
}

void RemoteTransformerBlockAtt::readWeights(char* woWeights) {
    size_t weightsBytes = this->woSlice->weights0Bytes;
    char* weights = new char[weightsBytes];

    this->woSlice->splitWeights(sliceIndex, woWeights, &weights[0]);

    client->createFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_ATT, weights, weightsBytes);
    delete[] weights;
}

void RemoteTransformerBlockAtt::beginForwarding() {
    state->sendUnitBuffer(SB_UNIT_XB, sliceIndex);
    client->forwardFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_ATT);
}

void RemoteTransformerBlockAtt::waitForEnd() {
    state->readSlicedBuffer(SB_SLICED_XB2, sliceIndex);
}

//
// TransformerBlockFfn
//

TransformerBlockFfn::TransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState *state)
    : TransformerFragment(layerIndex, sliceIndex, spec, state) {
    w1Slice = new MatMulSlice(spec->floatType, spec->sliceCount, spec->dim, spec->hiddenDim);
    w3Slice = new MatMulSlice(spec->floatType, spec->sliceCount, spec->dim, spec->hiddenDim);
}

TransformerBlockFfn::~TransformerBlockFfn() {
    delete w1Slice;
    delete w3Slice;
}

NativeTransformerBlockFfn::NativeTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState *state)
    : TransformerBlockFfn(layerIndex, sliceIndex, spec, state) {
    this->config = config;
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
    float* xb = (float*)state->getUnitBuffer(SB_UNIT_XB);
    float* hb0 = (float*)state->getSlicedBuffer(SB_SLICED_HB, sliceIndex);

    matmul(spec->floatType, config->nThread, hb0, xb, w1Weights0, w1Slice->n, w1Slice->d0);
    matmul(spec->floatType, config->nThread, hb20, xb, w3Weights0, w3Slice->n, w3Slice->d0);

    // SwiGLU non-linearity
    for (int i = 0; i < w1Slice->d0; i++) {
        float val = hb0[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= hb20[i];
        hb0[i] = val;
    }

    state->sendSlicedBuffer(SB_SLICED_HB, sliceIndex);
}

void NativeTransformerBlockFfn::waitForEnd() {}

RemoteTransformerBlockFfn::RemoteTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client)
    : TransformerBlockFfn(layerIndex, sliceIndex, spec, state) {
    this->client = client;
}

void RemoteTransformerBlockFfn::readWeights(char* w1Weights, char* w3Weights) {
    size_t weightsBytes = this->w1Slice->weights0Bytes + this->w3Slice->weights0Bytes;
    char* weights = new char[weightsBytes];

    this->w1Slice->splitWeights(sliceIndex, w1Weights, &weights[0]);
    this->w3Slice->splitWeights(sliceIndex, w3Weights, &weights[this->w1Slice->weights0Bytes]);

    client->createFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_FFN, weights, weightsBytes);

    delete[] weights;
}

void RemoteTransformerBlockFfn::beginForwarding() {
    state->sendUnitBuffer(SB_UNIT_XB, sliceIndex);
    client->forwardFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_FFN);
}

void RemoteTransformerBlockFfn::waitForEnd() {
    state->readSlicedBuffer(SB_SLICED_HB, sliceIndex);
}

//
// TransformerBlockFfn2
//

TransformerBlockFfn2::TransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState *state)
    : TransformerFragment(layerIndex, sliceIndex, spec, state) {
    w2Slice = new MatMulSlice(spec->floatType, spec->sliceCount, spec->hiddenDim, spec->dim);
}

TransformerBlockFfn2::~TransformerBlockFfn2() {
    delete w2Slice;
}

NativeTransformerBlockFfn2::NativeTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState *state)
    : TransformerBlockFfn2(layerIndex, sliceIndex, spec, state) {
    w2Weights0 = NEW_BUFFER(w2Slice->weights0Bytes);
    this->config = config;
}

NativeTransformerBlockFfn2::~NativeTransformerBlockFfn2() {
    FREE_BUFFER(w2Weights0);
}

void NativeTransformerBlockFfn2::readWeights(char *w2Weights) {
    this->w2Slice->splitWeights(sliceIndex, w2Weights, w2Weights0);
}

void NativeTransformerBlockFfn2::beginForwarding() {
    float *hh = (float*)state->getUnitBuffer(SB_UNIT_HH);
    float *xb2 = (float*)state->getSlicedBuffer(SB_SLICED_XB2, sliceIndex);

    matmul(spec->floatType, config->nThread, xb2, hh, w2Weights0, w2Slice->n, w2Slice->d0);

    state->sendSlicedBuffer(SB_SLICED_XB2, sliceIndex);
}

void NativeTransformerBlockFfn2::waitForEnd() {}

RemoteTransformerBlockFfn2::RemoteTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client)
    : TransformerBlockFfn2(layerIndex, sliceIndex, spec, state) {
    this->client = client;
}

void RemoteTransformerBlockFfn2::readWeights(char* w2Weights) {
    size_t weightsBytes = this->w2Slice->weights0Bytes;
    char* weights = new char[weightsBytes];

    this->w2Slice->splitWeights(sliceIndex, w2Weights, &weights[0]);

    client->createFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_FFN2, weights, weightsBytes);
    delete[] weights;
}

void RemoteTransformerBlockFfn2::beginForwarding() {
    state->sendUnitBuffer(SB_UNIT_HH, sliceIndex);
    client->forwardFragment(sliceIndex, layerIndex, TRANSFORMER_BLOCK_FFN2);
}

void RemoteTransformerBlockFfn2::waitForEnd() {
    state->readSlicedBuffer(SB_SLICED_XB2, sliceIndex);
}

//
// TransformerBlock
//

TransformerBlock::TransformerBlock(int layerIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState* state, RemoteClient* clientOrNull) {
    this->layerIndex = layerIndex;
    this->spec = spec;
    this->state = state;

    rmsAttWeight = new float[spec->dim];
    rmsFfnWeight = new float[spec->dim];

    qkvs = new TransformerBlockQkv*[spec->sliceCount];
    atts = new TransformerBlockAtt*[spec->sliceCount];
    ffns = new TransformerBlockFfn*[spec->sliceCount];
    ffn2s = new TransformerBlockFfn2*[spec->sliceCount];
    threadInfos = new TransformerBlockThreadInfo[spec->sliceCount];
    for (int s = 0; s < spec->sliceCount; s++) {
        if (clientOrNull == NULL) {
            qkvs[s] = new NativeTransformerBlockQkv(this->layerIndex, s, spec, config, state);
            atts[s] = new NativeTransformerBlockAtt(this->layerIndex, s, spec, config, state);
            ffns[s] = new NativeTransformerBlockFfn(this->layerIndex, s, spec, config, state);
            ffn2s[s] = new NativeTransformerBlockFfn2(this->layerIndex, s, spec, config, state);
        } else {
            qkvs[s] = new RemoteTransformerBlockQkv(this->layerIndex, s, spec, state, clientOrNull);
            atts[s] = new RemoteTransformerBlockAtt(this->layerIndex, s, spec, state, clientOrNull);
            ffns[s] = new RemoteTransformerBlockFfn(this->layerIndex, s, spec, state, clientOrNull);
            ffn2s[s] = new RemoteTransformerBlockFfn2(this->layerIndex, s, spec, state, clientOrNull);
        }
        TransformerBlockThreadInfo* ti = &threadInfos[s];
        ti->sliceIndex = s;
        ti->qkv = qkvs[s];
        ti->att = atts[s];
        ti->ffn = ffns[s];
        ti->ffn2 = ffn2s[s];
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
    delete[] threadInfos;

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
    char* w = wd;
    memcpy(rmsAttWeight, w, spec->dim * sizeof(float));
    w += spec->dim * sizeof(float);

    memcpy(rmsFfnWeight, w, spec->dim * sizeof(float));
    w += spec->dim * sizeof(float);

    char* wq = w;
    w += getBatchBytes(spec->floatType, spec->dim, spec->dim);
    char* wk = w;
    w += getBatchBytes(spec->floatType, spec->dim, spec->nKvHeads * spec->headSize);
    char* wv = w;
    w += getBatchBytes(spec->floatType, spec->dim, spec->nKvHeads * spec->headSize);

    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s]->readWeights(wq, wk, wv);
    }

    char* wo = w;
    w += getBatchBytes(spec->floatType, spec->dim, spec->dim);

    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->readWeights(wo);
    }

    char* w1 = w;
    w += getBatchBytes(spec->floatType, spec->dim, spec->hiddenDim);
    char* w2 = w;
    w += getBatchBytes(spec->floatType, spec->hiddenDim, spec->dim);
    char* w3 = w;
    w += getBatchBytes(spec->floatType, spec->dim, spec->hiddenDim);

    for (int s = 0; s < spec->sliceCount; s++) {
        ffns[s]->readWeights(w1, w3);
        ffn2s[s]->readWeights(w2);
    }
    return (long)(w - wd);
}

void* transformerBlockThread(void* arg) {
    TransformerBlockThreadInfo* info = (TransformerBlockThreadInfo*)arg;
    switch (info->step)
    {
        case TRANSFORMER_BLOCK_QKV:
        info->qkv->beginForwarding();
        info->qkv->waitForEnd();
        break;
        case TRANSFORMER_BLOCK_ATT:
        info->att->beginForwarding();
        info->att->waitForEnd();
        break;
        case TRANSFORMER_BLOCK_FFN:
        info->ffn->beginForwarding();
        info->ffn->waitForEnd();
        break;
        case TRANSFORMER_BLOCK_FFN2:
        info->ffn2->beginForwarding();
        info->ffn2->waitForEnd();
        break;
    }
    return 0;
}

void TransformerBlock::forward(int pos, float* x) {
    int dim = spec->dim;
    int kvDim = spec->kvDim;
    int kvMul = spec->nHeads / spec->nKvHeads; // integer multiplier of the kv sharing in multiquery
    int hiddenDim =  spec->hiddenDim;
    int headSize = dim / spec->nHeads;

    float* xb = (float*)state->getUnitBuffer(SB_UNIT_XB);
    float* hh = (float*)state->getUnitBuffer(SB_UNIT_HH);

    // attention rmsnorm
    rmsnorm(xb, x, rmsAttWeight, dim);

    // qkv matmuls for this position
    float* k = keyCache + pos * kvDim;
    float* v = valueCache + pos * kvDim;

    for (int s = 0; s < spec->sliceCount; s++) {
        threadInfos[s].step = TRANSFORMER_BLOCK_QKV;
        pthread_create(&threadInfos[s].handler, NULL, transformerBlockThread, (void*)&threadInfos[s]);
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        pthread_join(threadInfos[s].handler, NULL);
        TransformerBlockQkv *qkv = qkvs[s];
        qkv->qSlice->mergeOutputs(s, q, (float*)state->getSlicedBuffer(SB_SLICED_Q, s));
        qkv->kSlice->mergeOutputs(s, k, (float*)state->getSlicedBuffer(SB_SLICED_K, s));
        qkv->vSlice->mergeOutputs(s, v, (float*)state->getSlicedBuffer(SB_SLICED_V, s));
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
            float score = dotProduct(_q, k, headSize) / sqrtf(headSize);
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

    for (int s = 0; s < spec->sliceCount; s++) {
        threadInfos[s].step = TRANSFORMER_BLOCK_ATT;
        pthread_create(&threadInfos[s].handler, NULL, transformerBlockThread, (void*)&threadInfos[s]);
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        pthread_join(threadInfos[s].handler, NULL);
        atts[s]->woSlice->mergeOutputs(s, xb2, (float*)state->getSlicedBuffer(SB_SLICED_XB2, s));
    }

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
        x[i] += xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(xb, x, rmsFfnWeight, dim);

    for (int s = 0; s < spec->sliceCount; s++) {
        threadInfos[s].step = TRANSFORMER_BLOCK_FFN;
        pthread_create(&threadInfos[s].handler, NULL, transformerBlockThread, (void*)&threadInfos[s]);
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        pthread_join(threadInfos[s].handler, NULL);
        ffns[s]->w3Slice->mergeOutputs(s, hb, (float*)state->getSlicedBuffer(SB_SLICED_HB, s));
    }

    memcpy(hh, hb, hiddenDim * sizeof(float));

    for (int s = 0; s < spec->sliceCount; s++) {
        threadInfos[s].step = TRANSFORMER_BLOCK_FFN2;
        pthread_create(&threadInfos[s].handler, NULL, transformerBlockThread, (void*)&threadInfos[s]);
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        pthread_join(threadInfos[s].handler, NULL);
        ffn2s[s]->w2Slice->mergeOutputs(s, xb, (float*)state->getSlicedBuffer(SB_SLICED_XB2, s));
    }

    // residual connection
    for (int i = 0; i < dim; i++) {
        x[i] += xb[i];
    }
}

//
// Transformer
//

Transformer::Transformer(TransformerSpec* spec, TransformerConfig* config, TransformerState* state, RemoteClient* clientOrNull) {
    this->spec = spec;
    this->config = config;
    this->state = state;
    this->clientOrNull = clientOrNull;

    this->blocks = new TransformerBlock*[spec->nLayers];
    for (int l = 0; l < spec->nLayers; l++) {
        this->blocks[l] = new TransformerBlock(l, spec, config, state, clientOrNull);
    }

    x = new float[spec->dim];
    token_embedding_table = new float[spec->vocabSize * spec->dim];
    rms_final_weight = new float[spec->dim];

    wclsFloatType = spec->sharedWeights ? F32 : spec->floatType;
    wclsBytes = getBatchBytes(wclsFloatType, spec->vocabSize, spec->dim);
    wcls = new char[wclsBytes];

    logits = new float[spec->vocabSize];
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

long Transformer::readWeights(char* wd) {
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

    memcpy(wcls, spec->sharedWeights ? (char*)token_embedding_table : w, wclsBytes);
    if (!spec->sharedWeights) {
        w += wclsBytes;
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
    matmul(wclsFloatType, config->nThread, logits, x, wcls, spec->dim, spec->vocabSize);

    if (clientOrNull != NULL) {
        clientOrNull->dumpStatistics();
    }
}

void loadTransformerSpec(TransformerSpec* spec, const char* path, FloatType type, int sliceCount) {
    size_t headerBytes = 7 * sizeof(int);
    FILE* fh = fopen(path, "rb");
    if (fh == NULL) {
        printf("Cannot open file %s\n", path);
        exit(EXIT_FAILURE);
    }

    int config[7];
    fread(config, headerBytes, sizeof(char), fh);

    spec->dim = config[0];
    spec->hiddenDim = config[1];
    spec->nLayers = config[2];
    spec->nHeads = config[3];
    spec->nKvHeads = config[4];
    spec->sharedWeights = config[5] > 0 ? true : false;
    spec->vocabSize = abs(config[5]);
    spec->seqLen = config[6];
    spec->headSize = spec->dim / spec->nHeads;
    spec->kvDim = (spec->dim * spec->nKvHeads) / spec->nHeads;
    spec->floatType = type;
    spec->sliceCount = sliceCount;

    printf("dim: %d\n", spec->dim);
    printf("hiddenDim: %d\n", spec->hiddenDim);
    printf("nLayers: %d\n", spec->nLayers);
    printf("nHeads: %d\n", spec->nHeads);
    printf("nKvHeads: %d\n", spec->nKvHeads);
    printf("vocabSize: %d\n", spec->vocabSize);
    printf("seqLen: %d\n", spec->seqLen);

    fseek(fh, 0, SEEK_END);
    size_t fileSize = ftell(fh);
    fclose(fh);

    spec->fileSize = fileSize;
}

void loadTransformer(Transformer** transformerOut, TransformerSpec* spec, TransformerConfig* config, const char* path, RemoteClient* clientOrNull) {
    size_t headerBytes = 7 * sizeof(int);
    int fw = open(path, O_RDONLY);
    char* weights = (char*)mmap(NULL, spec->fileSize, PROT_READ, MAP_PRIVATE, fw, 0);
    if (weights == MAP_FAILED) {
        printf("Mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    weights += headerBytes;

    SharedBuffer* buffer = initSharedBuffer(spec);
    TransformerState* state = (clientOrNull == NULL)
        ? (TransformerState*)new NativeTransformerState(buffer)
        : (TransformerState*)new RemoteTransformerState(buffer, clientOrNull);
    Transformer* transformer = new Transformer(spec, config, state, clientOrNull);

    printf("Loading weights...\n");

    size_t loadedBytes = transformer->readWeights(weights);

    munmap(weights, spec->fileSize);

    size_t missedBytes = (headerBytes + loadedBytes) - spec->fileSize;
    if (missedBytes != 0) {
        printf("Missed %ld bytes\n", missedBytes);
        exit(EXIT_FAILURE);
    }

    printf("Loaded %ld bytes\n", loadedBytes);
    *transformerOut = transformer;
}
