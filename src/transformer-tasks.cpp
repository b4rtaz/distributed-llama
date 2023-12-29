#include <cmath>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "transformer-tasks.hpp"
#include <cassert>

#define TASK_ARGS unsigned int nThreads, unsigned int threadIndex, void* userData

#define TASK_VARIABLES \
    TransformerContext* ctx = (TransformerContext*)userData; \
    Transformer* transformer = ctx->transformer; \
    TransformerBlock* block = transformer->blocks[ctx->currentBlockIndex]; \
    TransformerSpec* spec = transformer->spec;

void syncUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    if (threadIndex != 0) return;

    char* buffer = ctx->transformer->buffer->getUnit(bufferIndex);

    size_t bufferBytes = ctx->transformer->buffer->getUnitBytes(bufferIndex);
    if (ctx->socketPool != NULL) {
        // root
        for (unsigned int socketIndex = 0; socketIndex < ctx->socketPool->nSockets; socketIndex++) {
            ctx->socketPool->write(socketIndex, buffer, bufferBytes);
        }
    } else if (ctx->socket != NULL) {
        // worker
        ctx->socket->read(buffer, bufferBytes);
    }
}

void syncSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    if (threadIndex != 0) return;

    size_t bufferBytes = ctx->transformer->buffer->getSlicedBytes(bufferIndex);
    if (ctx->socketPool != NULL) {
        // root
        for (unsigned int socketIndex = 0; socketIndex < ctx->socketPool->nSockets; socketIndex++) {
            uint8_t sliceIndex = socketIndex + 1;
            char* buffer = ctx->transformer->buffer->getSliced(bufferIndex, sliceIndex);
            ctx->socketPool->read(socketIndex, buffer, bufferBytes);
        }
    } else if (ctx->socket != NULL) {
        // worker
        char* buffer = ctx->transformer->buffer->getSliced(bufferIndex, ctx->transformer->sliceIndex);
        ctx->socket->write(buffer, bufferBytes);
    }
}

int rmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
    return TASK_CONTINUE;
}

int rmsAttNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)ctx->transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsAtt, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int syncRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB);
    return TASK_CONTINUE;
}

int qkv(TASK_ARGS) {
    TASK_VARIABLES;

    float *xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float *q0 = (float*)transformer->buffer->getSliced(TB_SLICED_Q, transformer->sliceIndex);
    float *k0 = (float*)transformer->buffer->getSliced(TB_SLICED_K, transformer->sliceIndex);
    float *v0 = (float*)transformer->buffer->getSliced(TB_SLICED_V, transformer->sliceIndex);

    matmul(spec->floatType, q0, xb, block->q0, block->q0Slice->n, block->q0Slice->d0, nThreads, threadIndex);
    matmul(spec->floatType, k0, xb, block->k0, block->k0Slice->n, block->k0Slice->d0, nThreads, threadIndex);
    matmul(spec->floatType, v0, xb, block->v0, block->v0Slice->n, block->v0Slice->d0, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int syncQkv(TASK_ARGS) {
    TASK_VARIABLES;
    syncSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_Q);
    syncSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_K);
    syncSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_V);

    // if (ctx->socketPool != NULL && threadIndex == 0) { float* v = (float*)block->q0; printf("q0 (%d): %f %f %f %f %f %f\n", ctx->currentBlockIndex, v[0], v[1], v[2], v[3], v[4], v[5]); }
    return TASK_CONTINUE;
}

int multiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex != 0) {
        return TASK_CONTINUE;
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
    return TASK_CONTINUE;
}

int syncMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB);
    return TASK_CONTINUE;
}

int att(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->floatType, xb2, xb, block->wo0, block->wo0Slice->n, block->wo0Slice->d0, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int syncAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2);
    return TASK_CONTINUE;
}

int rmfFfn(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
        float* x = (float*)transformer->x;

        for (int i = 0; i < spec->dim; i++) {
            x[i] += xb2[i];
        }
        transformer->rms = rms(x, spec->dim);
    }
    return TASK_CONTINUE;
}

int rmfFfnNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* x = (float*)transformer->x;

    rmsnorm(xb, x, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int syncRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB);
    return TASK_CONTINUE;
}

int ffn(TASK_ARGS) {
    TASK_VARIABLES;

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
    return TASK_CONTINUE;
}

int syncFfnA(TASK_ARGS) {
    TASK_VARIABLES;
    syncSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB);
    return TASK_CONTINUE;
}

int syncFfnB(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB);
    return TASK_CONTINUE;
}

int ffn2(TASK_ARGS) {
    TASK_VARIABLES;

    float *hb = (float*)transformer->buffer->getUnit(TB_SLICED_HB);
    float *xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->floatType, xb2, hb, block->w20, block->w20Slice->n, block->w20Slice->d0, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int syncFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    syncSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2);
    return TASK_CONTINUE;
}

int mergeFfn2(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0) {
        float* x = transformer->x;
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);

        for (int i = 0; i < spec->dim; i++) {
            x[i] += xb2[i];
        }
    }
    return TASK_CONTINUE;
}

int nextBlock(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0) {
        ctx->currentBlockIndex++;
        if (ctx->currentBlockIndex == spec->nLayers) {
            ctx->currentBlockIndex = 0;
            ctx->finalize = true;
        }
    }
    return TASK_CONTINUE;
}

int rmsFinal(TASK_ARGS) {
    TASK_VARIABLES;
    if (ctx->finalize && threadIndex == 0) {
        float* x = transformer->x;
        transformer->rms = rms(x, spec->dim);
    }
    return TASK_CONTINUE;
}

int rmsFinalNorm(TASK_ARGS) {
    TASK_VARIABLES;
    if (ctx->finalize) {
        float* x = transformer->x;
        rmsnorm(x, x, transformer->rms, (float*)transformer->rmsFinal, spec->dim, nThreads, threadIndex);
    }
    return TASK_CONTINUE;
}

int finalize(TASK_ARGS) {
    TASK_VARIABLES;

    if (ctx->finalize) {
        float* x = transformer->x;
        matmul(spec->floatType, transformer->logits, x, transformer->wcls, spec->dim, spec->vocabSize, nThreads, threadIndex);
        return TASK_STOP;
    }
    return TASK_CONTINUE;
}

static TaskLoopTask inferenceTasks[] = {
    { rmsAtt, TASK_TYPE_INFERENCE },
    { rmsAttNorm, TASK_TYPE_INFERENCE },
    { syncRmsAtt, TASK_TYPE_TRANSFER },
    { qkv, TASK_TYPE_INFERENCE },
    { syncQkv, TASK_TYPE_TRANSFER },
    { multiheadAtt, TASK_TYPE_INFERENCE },
    { syncMultiheadAtt, TASK_TYPE_TRANSFER },
    { att, TASK_TYPE_INFERENCE },
    { syncAtt, TASK_TYPE_TRANSFER },
    { rmfFfn, TASK_TYPE_INFERENCE },
    { rmfFfnNorm, TASK_TYPE_INFERENCE },
    { syncRmfFfn, TASK_TYPE_TRANSFER },
    { ffn, TASK_TYPE_INFERENCE },
    { syncFfnA, TASK_TYPE_TRANSFER },
    { syncFfnB, TASK_TYPE_TRANSFER },
    { ffn2, TASK_TYPE_INFERENCE },
    { syncFfn2, TASK_TYPE_TRANSFER },
    { mergeFfn2, TASK_TYPE_INFERENCE },
    { nextBlock, TASK_TYPE_INFERENCE },
    { rmsFinal, TASK_TYPE_INFERENCE },
    { rmsFinalNorm, TASK_TYPE_INFERENCE },
    { finalize, TASK_TYPE_INFERENCE },
};

TaskLoopTask* Inference::tasks = inferenceTasks;
int Inference::nTasks = sizeof(inferenceTasks) / sizeof(TaskLoopTask);

Inference::Inference(unsigned int nThreads, Transformer* transformer, SocketPool* socketPool) {
    this->transformer = transformer;
    context.transformer = transformer;
    context.socket = NULL;
    context.socketPool = socketPool;
    taskLoop = new TaskLoop(nThreads, nTasks, TASK_N_TYPES, tasks, (void*)&context);
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

void Inference::getStats(unsigned long* inferenceTime, unsigned long* transferTime) {
    *inferenceTime = taskLoop->executionTime[TASK_TYPE_INFERENCE];
    *transferTime = taskLoop->executionTime[TASK_TYPE_TRANSFER];
}

static TaskLoopTask workerTasks[] = {
    { syncRmsAtt, TASK_TYPE_TRANSFER },
    { qkv, TASK_TYPE_INFERENCE },
    { syncQkv, TASK_TYPE_TRANSFER },
    { syncMultiheadAtt, TASK_TYPE_TRANSFER },
    { att, TASK_TYPE_INFERENCE },
    { syncAtt, TASK_TYPE_TRANSFER },
    { syncRmfFfn, TASK_TYPE_TRANSFER },
    { ffn, TASK_TYPE_INFERENCE },
    { syncFfnA, TASK_TYPE_TRANSFER },
    { syncFfnB, TASK_TYPE_TRANSFER },
    { ffn2, TASK_TYPE_INFERENCE },
    { syncFfn2, TASK_TYPE_TRANSFER },
    { nextBlock, TASK_TYPE_INFERENCE },
};

TaskLoopTask* Worker::tasks = workerTasks;
int Worker::nTasks = sizeof(workerTasks) / sizeof(TaskLoopTask);

Worker::Worker(unsigned int nThreads, Transformer* transformer, Socket* socket) {
    this->transformer = transformer;
    context.transformer = transformer;
    context.socket = socket;
    context.socketPool = NULL;
    taskLoop = new TaskLoop(nThreads, nTasks, TASK_N_TYPES, tasks, (void*)&context);
}

Worker::~Worker() {
    delete taskLoop;
}

void Worker::work() {
    context.finalize = false;
    context.currentBlockIndex = 0;

    taskLoop->run();
}
