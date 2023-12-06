#include <math.h>
#include <string.h>
#include "shared-buffer.hpp"
#include "transformer.hpp"
#include "funcs.hpp"
#include "matmul.hpp"

TransformerBlockQkv::TransformerBlockQkv(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    qSlice = new MatMulSlice(spec->sliceCount, spec->dim, spec->dim);
    kSlice = new MatMulSlice(spec->sliceCount, spec->dim, spec->kv_dim);
    vSlice = new MatMulSlice(spec->sliceCount, spec->dim, spec->kv_dim);

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
    float *xb = (float*)sharedBuffer->get(SB_XB, 0);
    float *q0 = (float*)sharedBuffer->get(SB_Q, sliceIndex);
    float *k0 = (float*)sharedBuffer->get(SB_K, sliceIndex);
    float *v0 = (float*)sharedBuffer->get(SB_V, sliceIndex);

    matmul(q0, xb, qWeights0, qSlice->n, qSlice->d0);
    sharedBuffer->sync(1, sliceIndex);

    matmul(k0, xb, kWeights0, kSlice->n, kSlice->d0);
    sharedBuffer->sync(2, sliceIndex);

    matmul(v0, xb, vWeights0, vSlice->n, vSlice->d0);
    sharedBuffer->sync(3, sliceIndex);
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
    float *xb = (float*)sharedBuffer->get(SB_XB, 0);
    float *xb2 = (float*)sharedBuffer->get(SB_XB2, sliceIndex);

    matmul(xb2, xb, woWeights0, woSlice->n, woSlice->d0);
    sharedBuffer->sync(SB_XB2, sliceIndex);
}

void TransformerBlockAtt::waitForEnd() {
    // TODO
}

TransformerBlockFfn::TransformerBlockFfn(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;

    w1Slice = new MatMulSlice(spec->sliceCount, spec->hidden_dim, spec->dim);
    w2Slice = new MatMulSlice(spec->sliceCount, spec->dim, spec->hidden_dim);
    w3Slice = new MatMulSlice(spec->sliceCount, spec->hidden_dim, spec->dim);

    w1Weights0 = new float[w1Slice->weights0Length];
    w2Weights0 = new float[w2Slice->weights0Length];
    w3Weights0 = new float[w3Slice->weights0Length];
}

TransformerBlockFfn::~TransformerBlockFfn() {
    delete w1Slice;
    delete w2Slice;
    delete w3Slice;
    delete[] w1Weights0;
    delete[] w2Weights0;
    delete[] w3Weights0;
}

void TransformerBlockFfn::readWeights(float *w1Weights, float *w2Weights, float *w3Weights) {
    this->w1Slice->splitWeights(sliceIndex, w1Weights, w1Weights0);
    this->w2Slice->splitWeights(sliceIndex, w2Weights, w2Weights0);
    this->w3Slice->splitWeights(sliceIndex, w3Weights, w3Weights0);
}

void TransformerBlockFfn::beginForwarding() {
    float*xb = (float*)sharedBuffer->get(SB_XB, sliceIndex);

    matmul(hb0, xb, w1Weights0, w1Slice->n, w1Slice->d0);
    matmul(hb20, xb, w3Weights0, w3Slice->n, w3Slice->d0);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    // matmul(hb, xb, w1, dim, hidden_dim);
    // matmul(hb2, xb, w3, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < w1Slice->d0; i++) {
        float val = hb0[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= hb20[i];
        hb0[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(xb, hb0, w2Weights0, w2Slice->n, w2Slice->d0);
}

void TransformerBlockFfn::waitForEnd() {
    // TODO
}

TransformerBlock::TransformerBlock(
    TransformerSpec* spec,
    SharedBuffer* sharedBuffer,
    TransformerBlockQkv **qkvs,
    TransformerBlockAtt **atts) {
    this->spec = spec;
    this->sharedBuffer = sharedBuffer;
    this->qkvs = qkvs;
    this->atts = atts;

    rms_att_weight = new float[spec->dim];
    rms_ffn_weight = new float[spec->dim];
    //wo = new float[spec->n_heads * spec->head_size * spec->dim];
    w1 = new float[spec->hidden_dim * spec->dim];
    w2 = new float[spec->dim * spec->hidden_dim];
    w3 = new float[spec->hidden_dim * spec->dim];

    //xb = new float[spec->dim];
    xb2 = new float[spec->dim];
    hb = new float[spec->hidden_dim];
    hb2 = new float[spec->hidden_dim];
    q = new float[spec->dim];
    key_cache = new float[spec->seq_len * spec->kv_dim];
    value_cache = new float[spec->seq_len * spec->kv_dim];
    att = new float[spec->n_heads * spec->head_size];
}

TransformerBlock::~TransformerBlock() {
    delete[] rms_att_weight;
    delete[] rms_ffn_weight;
    //delete[] wo;
    delete[] w1;
    delete[] w2;
    delete[] w3;

    //delete[] xb;
    delete[] xb2;
    delete[] hb;
    delete[] hb2;
    delete[] q;
    delete[] key_cache;
    delete[] value_cache;
    delete[] att;
}

void TransformerBlock::readWeights(FILE *f) {
    fread(rms_att_weight, sizeof(float), spec->dim, f);
    fread(rms_ffn_weight, sizeof(float), spec->dim, f);
    
    float* wq = new float[spec->dim * spec->dim];
    float* wk = new float[spec->dim * spec->n_kv_heads * spec->head_size];
    float* wv = new float[spec->dim * spec->n_kv_heads * spec->head_size];
    fread(wq, sizeof(float), spec->dim * spec->dim, f);
    fread(wk, sizeof(float), spec->dim * spec->n_kv_heads * spec->head_size, f);
    fread(wv, sizeof(float), spec->dim * spec->n_kv_heads * spec->head_size, f);
    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s]->readWeights(wq, wk, wv);
    }
    delete[] wq;
    delete[] wk;
    delete[] wv;

    float* wo = new float[spec->dim * spec->dim];
    fread(wo, sizeof(float), spec->dim * spec->dim, f);
    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->readWeights(wo);
    }
    delete[] wo;

    // fread(wo, sizeof(float), spec->dim * spec->dim, f);
    fread(w1, sizeof(float), spec->hidden_dim * spec->dim, f);
    fread(w2, sizeof(float), spec->hidden_dim * spec->dim, f);
    fread(w3, sizeof(float), spec->hidden_dim * spec->dim, f);
}

void TransformerBlock::forward(int pos) {
    int dim = spec->dim;
    int kv_dim = spec->kv_dim;
    int kv_mul = spec->n_heads / spec->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  spec->hidden_dim;
    int head_size = dim / spec->n_heads;

    float* x = (float*)sharedBuffer->get(SB_X, 0);
    float* xb = (float*)sharedBuffer->get(SB_XB, 0);


    // attention rmsnorm
    float xss1 = sumOfSquares(x, dim);
    rmsnorm(xb, x, rms_att_weight, dim, xss1);

    printf("x: %f %f %f %f\n", x[0], x[1], x[2], x[3]);
    printf("xb: %f %f %f %f\n", xb[0], xb[1], xb[2], xb[3]);

    float* k = key_cache + pos * kv_dim;
    float* v = value_cache + pos * kv_dim;

    // qkv matmuls for this position
    memcpy(sharedBuffer->get(SB_XB, 0), xb, dim * sizeof(float));
    sharedBuffer->sync(SB_XB, 0);

    for (int s = 0; s < spec->sliceCount; s++) {
        qkvs[s]->beginForwarding();
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        TransformerBlockQkv *qkv = qkvs[s];
        qkv->waitForEnd();
        qkv->qSlice->mergeOutputs(s, q, (float*)sharedBuffer->get(SB_Q, s));
        qkv->kSlice->mergeOutputs(s, k, (float*)sharedBuffer->get(SB_K, s));
        qkv->vSlice->mergeOutputs(s, v, (float*)sharedBuffer->get(SB_V, s));
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
    for (h = 0; h < spec->n_heads; h++) {
        // get the query vector for this head
        float* _q = q + h * head_size;
        // attention scores for this head
        float* _att = att + h * spec->seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = key_cache + t * kv_dim + (h / kv_mul) * head_size;
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
            float* _v = value_cache + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = _att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                _xb[i] += a * _v[i];
            }
        }
    }

    memcpy(sharedBuffer->get(SB_XB, 0), xb, dim * sizeof(float));
    sharedBuffer->sync(SB_XB, 0);

    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->beginForwarding();
    }
    for (int s = 0; s < spec->sliceCount; s++) {
        atts[s]->waitForEnd();
        atts[s]->woSlice->mergeOutputs(s, xb2, (float*)sharedBuffer->get(SB_XB2, s));
    }

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
        x[i] += xb2[i];
    }

    // ffn rmsnorm
    float xss2 = sumOfSquares(x, dim);
    rmsnorm(xb, x, rms_ffn_weight, dim, xss2);

    sharedBuffer->sync(SB_XB, 0);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(hb, xb, w1, dim, hidden_dim);
    matmul(hb2, xb, w3, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
        float val = hb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= hb2[i];
        hb[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(xb, hb, w2, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
        x[i] += xb[i];
    }
}
