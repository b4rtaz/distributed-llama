#include <cmath>
#include <cassert>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"

void grokMulInput(TASK_ARGS) {
    TASK_VARIABLES;
    mulScalar(transformer->x, 78.38367176906169f, transformer->spec->dim, nThreads, threadIndex);
}

// source: https://github.com/karpathy/llama2.c/pull/408
void ropeFalcon(float* q, float* k, TransformerSpec* spec, int pos, float theta) {
    for (int i = 0; i < spec->nHeads; i++) {
        for (int j = 0; j < spec->headSize / 2; j++) {
            float freq = 1.0f / powf(theta, 2.0f * (float)j / (float)spec->headSize);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float q0 = q[i * spec->headSize + j];
            float q1 = q[i * spec->headSize + j + spec->headSize / 2];
            q[i * spec->headSize + j] = q0 * fcr - q1 * fci;
            q[i * spec->headSize + j + spec->headSize / 2] = q0 * fci + q1 * fcr;
            if (i < spec->nKvHeads) {
                float k0 = k[i * spec->headSize + j];
                float k1 = k[i * spec->headSize + j + spec->headSize / 2];
                k[i * spec->headSize + j] = k0 * fcr - k1 * fci;
                k[i * spec->headSize + j + spec->headSize / 2] = k0 * fci + k1 * fcr;
            }
        }
    }
}

void grokMultiheadAttRope(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* q = (float*)transformer->buffer->getUnit(TB_SLICED_Q);    
        float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
        float* k = block->keyCache + transformer->pos * spec->kvDim;
        ropeFalcon(q, k, spec, transformer->pos, spec->ropeTheta);
    }
}

void grokRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        transformer->rms = rms(xb2, spec->dim);
    }
}

void grokRmfFfnNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);

    rmsnorm(xb2, xb2, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
}

void grokRmfFfnNormJoin(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
}

void grokMoeRms(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
}

void grokMoeRmsNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsMoe, spec->dim, nThreads, threadIndex);
}

void grokMoeRouter(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    matmul(spec->weightsFloatType, F32, block->moeRouterProbs, xb, block->moeRouter, spec->dim, spec->nExperts, nThreads, threadIndex);
}

void grokMoeRouterSoftmax(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        softmax(block->moeRouterProbs, spec->nExperts);
    }
}

void grokMoeTopk(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        assert(spec->nActiveExperts == 2); // TODO
        uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);

        int best0i = -1;
        int best1i = -1;
        float best0v;
        float best1v;
        for (int i = 0; i < spec->nExperts; i++) {
            float prob = block->moeRouterProbs[i];
            if (best0i == -1 || best0v < prob) {
                if ((best0i != -1 && best1i == -1) || best1v < best0v) {
                    best1v = best0v;
                    best1i = best0i;
                }
                best0i = i;
                best0v = prob;
            } else if (best1i == -1 || best1v < prob) {
                best1i = i;
                best1v = prob;
            }
        }

        indexes[0] = (uint8_t)best0i;
        indexes[1] = (uint8_t)best1i;
    }
}

void grokMoeNormWeights(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);
        float* weights = (float*)transformer->buffer->getUnit(TB_UNIT_MOE_WEIGHTS);

        float sum = 0.0;
        int i;
        for (i = 0; i < spec->nActiveExperts; i++) {
            sum += block->moeRouterProbs[indexes[i]];
        }
        for (i = 0; i < spec->nActiveExperts; i++) {
            weights[i] = block->moeRouterProbs[indexes[i]] / sum;
        }
    }
}

void grokQuantizeMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
}

void grokSyncMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_INDEXES);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_WEIGHTS);
}

void grokMoeBlock0(TASK_ARGS) {
    TASK_VARIABLES;

    uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* hb = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        uint8_t e = indexes[ae];

        float* expertUp = &hb[block->moeUpAndGate0Slice->d0 * ae];
        float* expertGate = &block->expertGate[block->moeUpAndGate0Slice->d0 * ae];
        matmul(spec->weightsFloatType, spec->bufferFloatType, expertUp, xb, block->moeUp[e], block->moeUpAndGate0Slice->n, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
        matmul(spec->weightsFloatType, spec->bufferFloatType, expertGate, xb, block->moeGate[e], block->moeUpAndGate0Slice->n, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
    }
}

void grokMoeBlock1(TASK_ARGS) {
    TASK_VARIABLES;
    float* hb = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        float* expertUp = &hb[block->moeUpAndGate0Slice->d0 * ae];
        float* expertGate = &block->expertGate[block->moeUpAndGate0Slice->d0 * ae];

        if (spec->hiddenAct == SILU) {
            silu(expertGate, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
        } else if (spec->hiddenAct == GELU) {
            gelu(expertGate, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
        }
        mul(expertUp, expertGate, block->moeUpAndGate0Slice->d0, nThreads, threadIndex);
    }
}

void grokQuantizeMoeMul(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_SLICED_HB, TB_SLICED_HB_QUANTIZED);
}

void grokSyncMoeMulA(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
}

void grokSyncMoeMulRearrange(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0 && spec->nSlices > 1) {
        char* hbq = transformer->buffer->getUnit(TB_SLICED_HB_QUANTIZED);
        size_t bufferBytes = transformer->buffer->getUnitBytes(TB_SLICED_HB_QUANTIZED);
        size_t bufferSliceBytes = transformer->buffer->getSlicedBytes(TB_SLICED_HB_QUANTIZED);

        size_t moeUpBytes = bufferBytes / spec->nActiveExperts;
        size_t moeUp0SliceBytes = getBatchBytes(spec->bufferFloatType, block->moeUpAndGate0Slice->d0, 1);

        char* buffer = new char[bufferBytes];

        for (int s = 0; s < spec->nSlices; s++) {
            for (int ae = 0; ae < spec->nActiveExperts; ae++) {
                memcpy(&buffer[ae * moeUpBytes + s * moeUp0SliceBytes], &hbq[s * bufferSliceBytes + ae * moeUp0SliceBytes], moeUp0SliceBytes);
            }
        }

        memcpy(hbq, buffer, bufferBytes);
        delete[] buffer;
    }
}

void grokSyncMoeMulB(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
}

void grokMoeBlock2(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);
    char* hbq = transformer->buffer->getUnit(TB_SLICED_HB_QUANTIZED);
    size_t rowBytes = getBatchBytes(spec->bufferFloatType, spec->hiddenDim, 1);

    uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);
    float* weights = (float*)transformer->buffer->getUnit(TB_UNIT_MOE_WEIGHTS);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        uint8_t e = indexes[ae];
        float weight = weights[ae];

        char* expertUp = &hbq[rowBytes * ae];
        float* expertDown = ae == 0 ? xb2 : &block->expertDown[block->moeDown0Slice->d0 * (ae - 1)];

        matmul(spec->weightsFloatType, spec->bufferFloatType, expertDown, expertUp, block->moeDown[e], block->moeDown0Slice->n, block->moeDown0Slice->d0, nThreads, threadIndex);

        mulScalar(expertDown, weight, block->moeDown0Slice->d0, nThreads, threadIndex);
        if (ae > 0) {
            add(xb2, expertDown, block->moeDown0Slice->d0, nThreads, threadIndex);
        }
    }
}

void grokQuantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
}

void grokSyncMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
}

void grokDequantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);
}

void grokMoeRmsFinal(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        transformer->rms = rms(xb2, spec->dim);
    }
}

void grokMoeRmsNormFinal(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    rmsnorm(xb2, xb2, transformer->rms, block->rmsFfn2, spec->dim, nThreads, threadIndex);
}

void grokMoeAdd(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
}

void grokFinalize(TASK_ARGS) {
    TASK_VARIABLES;
    matmul(spec->weightsFloatType, F32, transformer->logits, transformer->x, transformer->wcls, spec->dim, spec->vocabSize, nThreads, threadIndex);
}

void grokFinalize2(TASK_ARGS) {
    TASK_VARIABLES;
    mulScalar(transformer->logits, 0.5773502691896257f, spec->vocabSize, nThreads, threadIndex);
}

TransformerArch buildGrok1Arch(TransformerSpec* spec) {
    TransformerArch a;

    // inference

    a.I(sendPoke, TASK_TYPE_TRANSFER);
    a.I(grokMulInput, TASK_TYPE_INFERENCE);
    for (int i = 0; i < spec->nLayers; i++) {
        a.I(llamaRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaRmsAttNorm, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.I(llamaAttQ, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAttQ, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAttQ, TASK_TYPE_INFERENCE);
        a.I(llamaAttK, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAttK, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAttK, TASK_TYPE_INFERENCE);
        a.I(llamaAttV, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAttV, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAttV, TASK_TYPE_INFERENCE);
        a.I(llamaDequantizeQkv, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(grokMultiheadAttRope, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAttJoin, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER);
        a.I(llamaAtt, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAtt, TASK_TYPE_TRANSFER);
        a.I(llamaDequantizeAtt, TASK_TYPE_INFERENCE);
        a.I(grokRmfFfn, TASK_TYPE_INFERENCE);
        a.I(grokRmfFfnNorm, TASK_TYPE_INFERENCE);
        a.I(grokRmfFfnNormJoin, TASK_TYPE_INFERENCE);

        a.I(grokMoeRms, TASK_TYPE_INFERENCE);
        a.I(grokMoeRmsNorm, TASK_TYPE_INFERENCE);
        a.I(grokMoeRouter, TASK_TYPE_INFERENCE);
        a.I(grokMoeRouterSoftmax, TASK_TYPE_INFERENCE);
        a.I(grokMoeTopk, TASK_TYPE_INFERENCE);
        a.I(grokMoeNormWeights, TASK_TYPE_INFERENCE);
        a.I(grokQuantizeMoeInput, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeInput, TASK_TYPE_TRANSFER);
        a.I(grokMoeBlock0, TASK_TYPE_INFERENCE);
        a.I(grokMoeBlock1, TASK_TYPE_INFERENCE);
        a.I(grokQuantizeMoeMul, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeMulA, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeMulRearrange, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeMulB, TASK_TYPE_INFERENCE);
        a.I(grokMoeBlock2, TASK_TYPE_INFERENCE);
        a.I(grokQuantizeMoeOutput, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeOutput, TASK_TYPE_TRANSFER);
        a.I(grokDequantizeMoeOutput, TASK_TYPE_INFERENCE);
        a.I(grokMoeRmsFinal, TASK_TYPE_INFERENCE);
        a.I(grokMoeRmsNormFinal, TASK_TYPE_INFERENCE);
        a.I(grokMoeAdd, TASK_TYPE_INFERENCE);
        a.I(llamaNextBlock, TASK_TYPE_INFERENCE);
    }

    a.I(llamaRmsFinal, TASK_TYPE_INFERENCE);
    a.I(llamaRmsFinalNorm, TASK_TYPE_INFERENCE);
    a.I(grokFinalize, TASK_TYPE_INFERENCE);
    a.I(grokFinalize2, TASK_TYPE_INFERENCE);

    // worker

    for (int i = 0; i < spec->nLayers; i++) {
        a.W(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.W(llamaAttQ, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAttQ, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAttQ, TASK_TYPE_INFERENCE);
        a.W(llamaAttK, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAttK, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAttK, TASK_TYPE_INFERENCE);
        a.W(llamaAttV, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAttV, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAttV, TASK_TYPE_INFERENCE);
        a.W(llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER);
        a.W(llamaAtt, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAtt, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAtt, TASK_TYPE_TRANSFER);

        a.W(grokSyncMoeInput, TASK_TYPE_TRANSFER);
        a.W(grokMoeBlock0, TASK_TYPE_INFERENCE);
        a.W(grokMoeBlock1, TASK_TYPE_INFERENCE);
        a.W(grokQuantizeMoeMul, TASK_TYPE_INFERENCE);
        a.W(grokSyncMoeMulA, TASK_TYPE_INFERENCE);
        a.W(grokSyncMoeMulB, TASK_TYPE_INFERENCE);
        a.W(grokMoeBlock2, TASK_TYPE_INFERENCE);
        a.W(grokQuantizeMoeOutput, TASK_TYPE_INFERENCE);
        a.W(grokSyncMoeOutput, TASK_TYPE_TRANSFER);

        a.W(llamaNextBlock, TASK_TYPE_INFERENCE);
    }

    return a;
}
