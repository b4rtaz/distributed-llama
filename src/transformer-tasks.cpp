#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "transformer-tasks.hpp"

#define TRANSFORMER_VARIABLES \
    TransformerContext* ctx = (TransformerContext*)userData; \
    Transformer* transformer = ctx->transformer; \
    TransformerBlock* block = transformer->blocks[ctx->currentBlockIndex]; \
    TransformerSpec* spec = transformer->spec;

int blockRmsAtt(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;
    if (threadIndex == 0) {
        float* xb = (float*)ctx->transformer->buffer->getUnit(TB_UNIT_XB);
        rmsnorm(xb, transformer->x, block->rmsAtt, spec->dim);
    }
    return TASK_LOOP_CONTINUE;
}

int blockQkv(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    float *xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float *q0 = (float*)transformer->buffer->getSliced(TB_SLICED_Q, transformer->sliceIndex);
    float *k0 = (float*)transformer->buffer->getSliced(TB_SLICED_K, transformer->sliceIndex);
    float *v0 = (float*)transformer->buffer->getSliced(TB_SLICED_V, transformer->sliceIndex);

    matmul(spec->floatType, q0, xb, block->q0, block->q0Slice->n, block->q0Slice->d0, nThreads, threadIndex);
    matmul(spec->floatType, k0, xb, block->k0, block->k0Slice->n, block->k0Slice->d0, nThreads, threadIndex);
    matmul(spec->floatType, v0, xb, block->v0, block->v0Slice->n, block->v0Slice->d0, nThreads, threadIndex);

    return TASK_LOOP_CONTINUE;
}

int blockMergeQkv(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;
    // TODO
    return TASK_LOOP_CONTINUE;
}

int blockMultiheadAtt(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;
    if (threadIndex != 0) {
        return TASK_LOOP_CONTINUE;
    }
    
    int dim = spec->dim;
    int kvDim = spec->kvDim;
    int kvMul = spec->nHeads / spec->nKvHeads; // integer multiplier of the kv sharing in multiquery
    int hiddenDim =  spec->hiddenDim;
    int headSize = dim / spec->nHeads;
    int pos = transformer->pos;

    float* q = (float*)transformer->buffer->getUnit(TB_SLICED_Q);    
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* k = block->keyCache + pos * kvDim;
    float* v = block->valueCache + pos * kvDim;

    memcpy(k, transformer->buffer->getUnit(TB_SLICED_K), dim * sizeof(float));
    memcpy(v, transformer->buffer->getUnit(TB_SLICED_V), dim * sizeof(float));

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
        float* _att = block->att + h * spec->seqLen;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = block->keyCache + t * kvDim + (h / kvMul) * headSize;
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
            float* _v = block->valueCache + t * kvDim + (h / kvMul) * headSize;
            // get the attention weight for this timestep
            float a = _att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < headSize; i++) {
                _xb[i] += a * _v[i];
            }
        }
    }
    return TASK_LOOP_CONTINUE;
}

int blockAtt(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    float *xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float *xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->floatType, xb2, xb, block->wo0, block->wo0Slice->n, block->wo0Slice->d0, nThreads, threadIndex);
    return TASK_LOOP_CONTINUE;
}

int blockMergeAtt(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    if (threadIndex == 0) {
        // TODO: merge xb2

        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
        float* x = (float*)transformer->x;

        for (int i = 0; i < spec->dim; i++) {
            x[i] += xb2[i];
        }

        rmsnorm(xb, x, block->rmsFfn, spec->dim);
    }
    return TASK_LOOP_CONTINUE;
}

int blockFfn(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* hb0 = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    matmul(spec->floatType, hb0, xb, block->w10, block->w10Slice->n, block->w10Slice->d0, nThreads, threadIndex);
    matmul(spec->floatType, block->hb20, xb, block->w30, block->w30Slice->n, block->w30Slice->d0, nThreads, threadIndex);

    // SwiGLU non-linearity
    int d00 = block->w10Slice->d0 / nThreads;
    int d0Offset = d00 * threadIndex;
    for (int i = 0; i < d00; i++) {
        float val = hb0[i + d0Offset];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= block->hb20[i + d0Offset];
        hb0[i + d0Offset] = val;
    }
    return TASK_LOOP_CONTINUE;
}

int blockMergeFfn(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    if (threadIndex == 0) {
        float* hb = (float*)transformer->buffer->getUnit(TB_SLICED_HB);
        float* hh = (float*)transformer->buffer->getUnit(TB_UNIT_HH);

        memcpy(hh, hb, spec->hiddenDim * sizeof(float));
    }
    return TASK_LOOP_CONTINUE;
}

int blockFfn2(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    float *hh = (float*)transformer->buffer->getUnit(TB_UNIT_HH);
    float *xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->floatType, xb2, hh, block->w20, block->w20Slice->n, block->w20Slice->d0, nThreads, threadIndex);
    return TASK_LOOP_CONTINUE;
}

int blockMergeFfn2(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    if (threadIndex == 0) {
        float* x = transformer->x;
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);

        for (int i = 0; i < spec->dim; i++) {
            x[i] += xb2[i];
        }
    }
    return TASK_LOOP_CONTINUE;
}

int nextBlock(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    if (threadIndex == 0) {
        ctx->currentBlockIndex++;

        if (ctx->currentBlockIndex == spec->nLayers) {
            float* x = transformer->x;
            rmsnorm(x, x, (float*)transformer->rmsFinal, spec->dim);
            ctx->finalize = true;
        }
    }
    return TASK_LOOP_CONTINUE;
}

int finalize(unsigned int nThreads, unsigned int threadIndex, void* userData) {
    TRANSFORMER_VARIABLES;

    if (ctx->finalize) {
        float* x = transformer->x;
        matmul(spec->floatType, transformer->logits, x, transformer->wcls, spec->dim, spec->vocabSize, nThreads, threadIndex);
        return TASK_LOOP_STOP;
    }

    return TASK_LOOP_CONTINUE;
}

static TaskLoopTask inferenceTasks[] = {
    blockRmsAtt,
    blockQkv,
    blockMergeQkv,
    blockMultiheadAtt,
    blockAtt,
    blockMergeAtt,
    blockFfn,
    blockMergeFfn,
    blockFfn2,
    blockMergeFfn2,
    nextBlock,
    finalize,
};

TaskLoopTask* Inference::tasks = inferenceTasks;
int Inference::nTasks = sizeof(inferenceTasks) / sizeof(TaskLoopTask);

Inference::Inference(unsigned int nThreads, Transformer* transformer) {
    this->transformer = transformer;
    context.transformer = transformer;
    taskLoop = new TaskLoop(nThreads, nTasks, tasks, (void*)&context);
}

Inference::~Inference() {
    delete taskLoop;
}

float* Inference::infer(int token, int pos) {
    transformer->pos = pos;

    float* contentRow = ((float*)transformer->tokenEmbeddingTable) + token * transformer->spec->dim;
    memcpy(transformer->x, contentRow, transformer->spec->dim * sizeof(float));

    context.finalize = false;
    context.currentBlockIndex = 0;

    taskLoop->run();

    return transformer->logits;
}
