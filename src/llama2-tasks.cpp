#include <cmath>
#include <cassert>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"

int llamaRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
    return TASK_CONTINUE;
}

int llamaRmsAttNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsAtt, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int llamaQuantizeRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaQkv(TASK_ARGS) {
    TASK_VARIABLES;

    float *xbq = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float *q0 = (float*)transformer->buffer->getSliced(TB_SLICED_Q, transformer->sliceIndex);
    float *k0 = (float*)transformer->buffer->getSliced(TB_SLICED_K, transformer->sliceIndex);
    float *v0 = (float*)transformer->buffer->getSliced(TB_SLICED_V, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, q0, xbq, block->q0, block->q0Slice->n, block->q0Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, k0, xbq, block->k0, block->k0Slice->n, block->k0Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, v0, xbq, block->v0, block->v0Slice->n, block->v0Slice->d0, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int llamaQuantizeQkv(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_Q, TB_SLICED_Q_QUANTIZED);
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_K, TB_SLICED_K_QUANTIZED);
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_V, TB_SLICED_V_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncQkv(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_Q_QUANTIZED);
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_K_QUANTIZED);
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_V_QUANTIZED);
    // if (ctx->socketPool != NULL && threadIndex == 0) { float* v = (float*)block->q0; printf("q0 (%d): %f %f %f %f %f %f\n", ctx->currentBlockIndex, v[0], v[1], v[2], v[3], v[4], v[5]); }
    return TASK_CONTINUE;
}

int llamaDequantizeQkv(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_Q_QUANTIZED, TB_SLICED_Q);
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_K_QUANTIZED, TB_SLICED_K);
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_V_QUANTIZED, TB_SLICED_V);
    return TASK_CONTINUE;
}

int llamaMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* k = block->keyCache + transformer->pos * spec->kvDim;
        float* v = block->valueCache + transformer->pos * spec->kvDim;

        memcpy(k, transformer->buffer->getUnit(TB_SLICED_K), spec->dim * sizeof(float));
        memcpy(v, transformer->buffer->getUnit(TB_SLICED_V), spec->dim * sizeof(float));
    }
    return TASK_CONTINUE;
}

int llamaMultiheadAttRope(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* q = (float*)transformer->buffer->getUnit(TB_SLICED_Q);
        float* k = block->keyCache + transformer->pos * spec->kvDim;

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < spec->dim; i+=2) {
            int head_dim = i % spec->headSize;
            float freq = 1.0f / powf(spec->ropeTheta, head_dim / (float)spec->headSize);
            float val = transformer->pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < spec->kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int _v = 0; _v < rotn; _v++) {
                float* vec = _v == 0 ? q : k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
    }
    return TASK_CONTINUE;
}

int llamaMultiheadAttJoin(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* q = (float*)transformer->buffer->getUnit(TB_SLICED_Q);
        float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
        int kvMul = spec->nHeads / spec->nKvHeads; // integer multiplier of the kv sharing in multiquery

        // multihead attention. iterate over all heads
        int h;
        for (h = 0; h < spec->nHeads; h++) {
            // get the query vector for this head
            float* _q = q + h * spec->headSize;
            // attention scores for this head
            float* _att = block->att + h * spec->seqLen;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= transformer->pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = block->keyCache + t * spec->kvDim + (h / kvMul) * spec->headSize;
                // calculate the attention score as the dot product of q and k
                float score = dotProduct(_q, k, spec->headSize) / sqrtf(spec->headSize);
                _att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(_att, transformer->pos + 1);

            // weighted sum of the values, store back into xb
            float* _xb = xb + h * spec->headSize;
            memset(_xb, 0, spec->headSize * sizeof(float));
            for (int t = 0; t <= transformer->pos; t++) {
                // get the value vector for this head and at this timestep
                float* _v = block->valueCache + t * spec->kvDim + (h / kvMul) * spec->headSize;
                // get the attention weight for this timestep
                float a = _att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < spec->headSize; i++) {
                    _xb[i] += a * _v[i];
                }
            }
        }
    }
    return TASK_CONTINUE;
}

int llamaQuantizeMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaAtt(TASK_ARGS) {
    TASK_VARIABLES;

    char* xb = transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, xb2, xb, block->wo0, block->wo0Slice->n, block->wo0Slice->d0, nThreads, threadIndex);

    return TASK_CONTINUE;
}

int llamaQuantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaDequantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);    
    return TASK_CONTINUE;
}

int llamaRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        float* x = (float*)transformer->x;

        for (int i = 0; i < spec->dim; i++) {
            x[i] += xb2[i];
        }
        transformer->rms = rms(x, spec->dim);
    }
    return TASK_CONTINUE;
}

int llamaRmfFfnNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* x = (float*)transformer->x;

    rmsnorm(xb, x, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int llamaQuantizeRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaFfn(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* hb0 = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, hb0, xb, block->w10, block->w10Slice->n, block->w10Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, block->hb20, xb, block->w30, block->w30Slice->n, block->w30Slice->d0, nThreads, threadIndex);

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

int llamaQuantizeFfnA(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_SLICED_HB, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncFfnA(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncFfnB(TASK_ARGS) {
    TASK_VARIABLES;
    syncMissingSlicesOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaFfn2(TASK_ARGS) {
    TASK_VARIABLES;

    float *hb = (float*)transformer->buffer->getUnit(TB_SLICED_HB_QUANTIZED);
    float *xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, xb2, hb, block->w20, block->w20Slice->n, block->w20Slice->d0, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int llamaQuantizeFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaSyncFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int llamaDequantizeFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);
    return TASK_CONTINUE;
}

int llamaMergeFfn2(TASK_ARGS) {
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

int llamaNextBlock(TASK_ARGS) {
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

int llamaRmsFinal(TASK_ARGS) {
    TASK_VARIABLES;
    if (ctx->finalize && threadIndex == 0) {
        float* x = transformer->x;
        transformer->rms = rms(x, spec->dim);
    }
    return TASK_CONTINUE;
}

int llamaRmsFinalNorm(TASK_ARGS) {
    TASK_VARIABLES;
    if (ctx->finalize) {
        float* x = transformer->x;
        rmsnorm(x, x, transformer->rms, (float*)transformer->rmsFinal, spec->dim, nThreads, threadIndex);
    }
    return TASK_CONTINUE;
}

int llamaFinalize(TASK_ARGS) {
    TASK_VARIABLES;

    if (ctx->finalize) {
        float* x = transformer->x;
        matmul(spec->weightsFloatType, F32, transformer->logits, x, transformer->wcls, spec->dim, spec->vocabSize, nThreads, threadIndex);
        return TASK_STOP;
    }
    return TASK_CONTINUE;
}

static TaskLoopTask inferenceTasks[] = {
    { llamaRmsAtt, TASK_TYPE_INFERENCE },
    { llamaRmsAttNorm, TASK_TYPE_INFERENCE },
    { llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE },
    { llamaSyncRmsAtt, TASK_TYPE_TRANSFER },
    { llamaQkv, TASK_TYPE_INFERENCE },
    { llamaQuantizeQkv, TASK_TYPE_INFERENCE },
    { llamaSyncQkv, TASK_TYPE_TRANSFER },
    { llamaDequantizeQkv, TASK_TYPE_INFERENCE },
    { llamaMultiheadAtt, TASK_TYPE_INFERENCE },
    { llamaMultiheadAttRope, TASK_TYPE_INFERENCE },
    { llamaMultiheadAttJoin, TASK_TYPE_INFERENCE },
    { llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE },
    { llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER },
    { llamaAtt, TASK_TYPE_INFERENCE },
    { llamaQuantizeAtt, TASK_TYPE_INFERENCE },
    { llamaSyncAtt, TASK_TYPE_TRANSFER },
    { llamaDequantizeAtt, TASK_TYPE_INFERENCE },
    { llamaRmfFfn, TASK_TYPE_INFERENCE },
    { llamaRmfFfnNorm, TASK_TYPE_INFERENCE },
    { llamaQuantizeRmfFfn, TASK_TYPE_INFERENCE },
    { llamaSyncRmfFfn, TASK_TYPE_TRANSFER },
    { llamaFfn, TASK_TYPE_INFERENCE },
    { llamaQuantizeFfnA, TASK_TYPE_INFERENCE },
    { llamaSyncFfnA, TASK_TYPE_TRANSFER },
    { llamaSyncFfnB, TASK_TYPE_TRANSFER },
    { llamaFfn2, TASK_TYPE_INFERENCE },
    { llamaQuantizeFfn2, TASK_TYPE_INFERENCE },
    { llamaSyncFfn2, TASK_TYPE_TRANSFER },
    { llamaDequantizeFfn2, TASK_TYPE_INFERENCE },
    { llamaMergeFfn2, TASK_TYPE_INFERENCE },
    { llamaNextBlock, TASK_TYPE_INFERENCE },
    { llamaRmsFinal, TASK_TYPE_INFERENCE },
    { llamaRmsFinalNorm, TASK_TYPE_INFERENCE },
    { llamaFinalize, TASK_TYPE_INFERENCE },
};

static TaskLoopTask workerTasks[] = {
    { llamaSyncRmsAtt, TASK_TYPE_TRANSFER },
    { llamaQkv, TASK_TYPE_INFERENCE },
    { llamaQuantizeQkv, TASK_TYPE_INFERENCE },
    { llamaSyncQkv, TASK_TYPE_TRANSFER },
    { llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER },
    { llamaAtt, TASK_TYPE_INFERENCE },
    { llamaQuantizeAtt, TASK_TYPE_INFERENCE },
    { llamaSyncAtt, TASK_TYPE_TRANSFER },
    { llamaSyncRmfFfn, TASK_TYPE_TRANSFER },
    { llamaFfn, TASK_TYPE_INFERENCE },
    { llamaQuantizeFfnA, TASK_TYPE_INFERENCE },
    { llamaSyncFfnA, TASK_TYPE_TRANSFER },
    { llamaSyncFfnB, TASK_TYPE_TRANSFER },
    { llamaFfn2, TASK_TYPE_INFERENCE },
    { llamaQuantizeFfn2, TASK_TYPE_INFERENCE },
    { llamaSyncFfn2, TASK_TYPE_TRANSFER },
    { llamaNextBlock, TASK_TYPE_INFERENCE },
};

TransformerArch Llama2::arch = {
    .initInference = NULL,
    .inference = {
        .nTasks = sizeof(inferenceTasks) / sizeof(TaskLoopTask),
        .tasks = inferenceTasks
    },
    .worker = {
        .nTasks = sizeof(workerTasks) / sizeof(TaskLoopTask),
        .tasks = workerTasks
    }
};
