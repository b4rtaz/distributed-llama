#include <cmath>
#include <cassert>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"

void llamaRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
}

void llamaRmsAttNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsAtt, spec->dim, nThreads, threadIndex);
}

void llamaQuantizeRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
}

void llamaSyncRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
}

void llamaQkv(TASK_ARGS) {
    TASK_VARIABLES;

    float *xbq = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float *q0 = (float*)transformer->buffer->getSliced(TB_SLICED_Q, transformer->sliceIndex);
    float *k0 = (float*)transformer->buffer->getSliced(TB_SLICED_K, transformer->sliceIndex);
    float *v0 = (float*)transformer->buffer->getSliced(TB_SLICED_V, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, q0, xbq, block->q0, block->q0Slice->n, block->q0Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, k0, xbq, block->k0, block->k0Slice->n, block->k0Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, v0, xbq, block->v0, block->v0Slice->n, block->v0Slice->d0, nThreads, threadIndex);
}

void llamaQuantizeQkv(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_Q, TB_SLICED_Q_QUANTIZED);
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_K, TB_SLICED_K_QUANTIZED);
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_V, TB_SLICED_V_QUANTIZED);
}

void llamaSyncQkv(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_Q_QUANTIZED);
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_K_QUANTIZED);
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_V_QUANTIZED);
    // if (ctx->socketPool != NULL && threadIndex == 0) { float* v = (float*)block->q0; printf("q0 (%d): %f %f %f %f %f %f\n", ctx->currentBlockIndex, v[0], v[1], v[2], v[3], v[4], v[5]); }
}

void llamaDequantizeQkv(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_Q_QUANTIZED, TB_SLICED_Q);
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_K_QUANTIZED, TB_SLICED_K);
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_V_QUANTIZED, TB_SLICED_V);
}

void llamaMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* k = block->keyCache + transformer->pos * spec->kvDim;
        float* v = block->valueCache + transformer->pos * spec->kvDim;

        memcpy(k, transformer->buffer->getUnit(TB_SLICED_K), spec->dim * sizeof(float));
        memcpy(v, transformer->buffer->getUnit(TB_SLICED_V), spec->dim * sizeof(float));
    }
}

void llamaMultiheadAttRope(TASK_ARGS) {
    TASK_VARIABLES;
    float* q = (float*)transformer->buffer->getUnit(TB_SLICED_Q);
    float* k = block->keyCache + transformer->pos * spec->kvDim;
    rope(transformer->ropeCache, q, k, spec, transformer->pos, nThreads, threadIndex);
}

void llamaMultiheadAttJoin(TASK_ARGS) {
    TASK_VARIABLES;
    float* q = (float*)transformer->buffer->getUnit(TB_SLICED_Q);
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);

    int kvMul = spec->nHeads / spec->nKvHeads; // integer multiplier of the kv sharing in multiquery
    int nHeadsPerThread = spec->nHeads / nThreads;

    int hStart = threadIndex * nHeadsPerThread;
    int hEnd = (threadIndex == nThreads - 1) ? spec->nHeads : (hStart + nHeadsPerThread);

    for (int h = hStart; h < hEnd; h++) {
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

void llamaQuantizeMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
}

void llamaSyncMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
}

void llamaAtt(TASK_ARGS) {
    TASK_VARIABLES;

    char* xb = transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, xb2, xb, block->wo0, block->wo0Slice->n, block->wo0Slice->d0, nThreads, threadIndex);

}

void llamaQuantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
}

void llamaSyncAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
}

void llamaDequantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);    
}

void llamaMergeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    float* x = (float*)transformer->x;
    add(x, xb2, spec->dim, nThreads, threadIndex);
}

void llamaRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
}

void llamaRmfFfnNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* x = (float*)transformer->x;

    rmsnorm(xb, x, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
}

void llamaQuantizeRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
}

void llamaSyncRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
}

void llamaFfn(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* hb0 = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, hb0, xb, block->w10, block->w10Slice->n, block->w10Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, block->hb20, xb, block->w30, block->w30Slice->n, block->w30Slice->d0, nThreads, threadIndex);

    silu(hb0, block->w10Slice->d0, nThreads, threadIndex);
    mul(hb0, block->hb20, block->w10Slice->d0, nThreads, threadIndex);
}

void llamaQuantizeFfnA(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_SLICED_HB, TB_SLICED_HB_QUANTIZED);
}

void llamaSyncFfnA(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
}

void llamaSyncFfnB(TASK_ARGS) {
    TASK_VARIABLES;
    syncMissingSlicesOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
}

void llamaFfn2(TASK_ARGS) {
    TASK_VARIABLES;

    float *hb = (float*)transformer->buffer->getUnit(TB_SLICED_HB_QUANTIZED);
    float *xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, xb2, hb, block->w20, block->w20Slice->n, block->w20Slice->d0, nThreads, threadIndex);
}

void llamaQuantizeFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
}

void llamaSyncFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
}

void llamaDequantizeFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);
}

void llamaMergeFfn2(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
}

void llamaNextBlock(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0) {
        ctx->currentBlockIndex++;
    }
}

void llamaRmsFinal(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* x = transformer->x;
        transformer->rms = rms(x, spec->dim);
    }
}

void llamaRmsFinalNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* x = transformer->x;
    rmsnorm(x, x, transformer->rms, (float*)transformer->rmsFinal, spec->dim, nThreads, threadIndex);
}

void llamaFinalize(TASK_ARGS) {
    TASK_VARIABLES;

    float* x = transformer->x;
    matmul(spec->weightsFloatType, F32, transformer->logits, x, transformer->wcls, spec->dim, spec->vocabSize, nThreads, threadIndex);
}

TransformerArch buildLlama2Arch(TransformerSpec* spec) {
    TransformerArch a;

    // inference

    a.I(sendPos, TASK_TYPE_TRANSFER);
    for (int i = 0; i < spec->nLayers; i++) {
        a.I(llamaRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaRmsAttNorm, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.I(llamaQkv, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeQkv, TASK_TYPE_INFERENCE);
        a.I(llamaSyncQkv, TASK_TYPE_TRANSFER);
        a.I(llamaDequantizeQkv, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAttRope, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAttJoin, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER);
        a.I(llamaAtt, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAtt, TASK_TYPE_TRANSFER);
        a.I(llamaDequantizeAtt, TASK_TYPE_INFERENCE);
        a.I(llamaMergeAtt, TASK_TYPE_INFERENCE);
        a.I(llamaRmfFfn, TASK_TYPE_INFERENCE);
        a.I(llamaRmfFfnNorm, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeRmfFfn, TASK_TYPE_INFERENCE);
        a.I(llamaSyncRmfFfn, TASK_TYPE_TRANSFER);
        a.I(llamaFfn, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeFfnA, TASK_TYPE_INFERENCE);
        a.I(llamaSyncFfnA, TASK_TYPE_TRANSFER);
        a.I(llamaSyncFfnB, TASK_TYPE_TRANSFER);
        a.I(llamaFfn2, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeFfn2, TASK_TYPE_INFERENCE);
        a.I(llamaSyncFfn2, TASK_TYPE_TRANSFER);
        a.I(llamaDequantizeFfn2, TASK_TYPE_INFERENCE);
        a.I(llamaMergeFfn2, TASK_TYPE_INFERENCE);
        a.I(llamaNextBlock, TASK_TYPE_INFERENCE);
    }
    a.I(llamaRmsFinal, TASK_TYPE_INFERENCE);
    a.I(llamaRmsFinalNorm, TASK_TYPE_INFERENCE);
    a.I(llamaFinalize, TASK_TYPE_INFERENCE);

    // worker

    for (int i = 0; i < spec->nLayers; i++) {
        a.W(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.W(llamaQkv, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeQkv, TASK_TYPE_INFERENCE);
        a.W(llamaSyncQkv, TASK_TYPE_TRANSFER);
        a.W(llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER);
        a.W(llamaAtt, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAtt, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAtt, TASK_TYPE_TRANSFER);
        a.W(llamaSyncRmfFfn, TASK_TYPE_TRANSFER);
        a.W(llamaFfn, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeFfnA, TASK_TYPE_INFERENCE);
        a.W(llamaSyncFfnA, TASK_TYPE_TRANSFER);
        a.W(llamaSyncFfnB, TASK_TYPE_TRANSFER);
        a.W(llamaFfn2, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeFfn2, TASK_TYPE_INFERENCE);
        a.W(llamaSyncFfn2, TASK_TYPE_TRANSFER);
        a.W(llamaNextBlock, TASK_TYPE_INFERENCE);
    }
    return a;
}
