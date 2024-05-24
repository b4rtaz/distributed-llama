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
    assert(block->kvCacheSlice->kvDim0 == block->k0Slice->d0);
    assert(block->kvCacheSlice->kvDim0 == block->v0Slice->d0);

    float *xbq = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float *k0 = &block->kvCacheSlice->keyCache[transformer->pos * block->kvCacheSlice->kvDim0];
    float* v0 = &block->kvCacheSlice->valueCache[transformer->pos * block->kvCacheSlice->kvDim0];

    matmul(spec->weightsFloatType, spec->bufferFloatType, block->qo0, xbq, block->q0, block->q0Slice->n, block->q0Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, k0, xbq, block->k0, block->k0Slice->n, block->k0Slice->d0, nThreads, threadIndex);
    matmul(spec->weightsFloatType, spec->bufferFloatType, v0, xbq, block->v0, block->v0Slice->n, block->v0Slice->d0, nThreads, threadIndex);
}

void llamaRope(TASK_ARGS) {
    TASK_VARIABLES;
    float* k0 = &block->kvCacheSlice->keyCache[transformer->pos * block->kvCacheSlice->kvDim0];
    transformer->ropeSlice->forward(true, block->qo0, transformer->pos, nThreads, threadIndex);
    transformer->ropeSlice->forward(false, k0, transformer->pos, nThreads, threadIndex);
}

void llamaMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    SPLIT_RANGE_TO_THREADS(h0Start, h0End, 0, block->multiHeadAttSlice->nHeads0, nThreads, threadIndex);

    float* xb = (float*)transformer->buffer->getSliced(TB_UNIT_XB, transformer->sliceIndex);

    int kvMul = spec->nHeads / spec->nKvHeads; // integer multiplier of the kv sharing in multiquery

    for (int h0 = h0Start; h0 < h0End; h0++) {
        // get the query vector for this head
        float* _q = block->qo0 + h0 * spec->headSize;
        // attention scores for this head
        float* _att = block->multiHeadAttSlice->att + h0 * spec->seqLen;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= transformer->pos; t++) {
            // get the key vector for this head and at this timestep
            float* k = block->kvCacheSlice->keyCache + t * block->kvCacheSlice->kvDim0 + (h0 / kvMul) * spec->headSize;
            // calculate the attention score as the dot product of q and k
            float score = dotProduct(_q, k, spec->headSize) / sqrtf(spec->headSize);
            _att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(_att, transformer->pos + 1);

        // weighted sum of the values, store back into xb
        float* hxb = xb + h0 * spec->headSize;
        memset(hxb, 0, spec->headSize * sizeof(float));
        for (int t = 0; t <= transformer->pos; t++) {
            // get the value vector for this head and at this timestep
            float* _v = block->kvCacheSlice->valueCache + t * block->kvCacheSlice->kvDim0 + (h0 / kvMul) * spec->headSize;
            // get the attention weight for this timestep
            float a = _att[t];

            // accumulate the weighted value into xb
            for (int i = 0; i < spec->headSize; i++) {
                hxb[i] += a * _v[i];
            }
        }
    }
}

void llamaQuantizeMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
};

void llamaAtt(TASK_ARGS) {
    TASK_VARIABLES;

    char* xbq0 = transformer->buffer->getSliced(TB_UNIT_XB_QUANTIZED, transformer->sliceIndex);
    float* xbv0 = (float*)transformer->buffer->getSliced(TB_SLICED_XBV, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, xbv0, xbq0, block->wo0, block->wo0Slice->n0, block->wo0Slice->d, nThreads, threadIndex);
}

void llamaQuantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XBV, TB_SLICED_XBV_QUANTIZED);
}

void llamaSyncAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XBV_QUANTIZED);
}

void llamaDequantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XBV_QUANTIZED, TB_SLICED_XBV);    
}

void llamaMergeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    for (uint8_t sliceIndex = 0; sliceIndex < spec->nSlices; sliceIndex++) {
        float* xbv = (float*)transformer->buffer->getSliced(TB_SLICED_XBV, sliceIndex);
        add(transformer->x, xbv, spec->dim, nThreads, threadIndex);
    }
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

    if (spec->hiddenAct == SILU) {
        silu(hb0, block->w10Slice->d0, nThreads, threadIndex);
    } else if (spec->hiddenDim == GELU) {
        gelu(hb0, block->w10Slice->d0, nThreads, threadIndex);
    } else {
        assert(false);
    }
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

TransformerArch buildLlamaArch(TransformerSpec* spec) {
    TransformerArch a;

    // inference

    a.I(sendPos, TASK_TYPE_TRANSFER);
    for (int i = 0; i < spec->nLayers; i++) {
        a.I(llamaRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaRmsAttNorm, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.I(llamaQkv, TASK_TYPE_INFERENCE);
        a.I(llamaRope, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);
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
        a.W(llamaRope, TASK_TYPE_INFERENCE);
        a.W(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);
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
