#include <cmath>
#include <cassert>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"

void initInference(TransformerContext* context) {
    Transformer* transformer = context->transformer;
    mulScalar(transformer->x, 78.38367176906169f, transformer->spec->dim, 1, 0);
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

int grokMultiheadAttRope(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* q = (float*)transformer->buffer->getUnit(TB_SLICED_Q);    
        float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
        float* k = block->keyCache + transformer->pos * spec->kvDim;
        ropeFalcon(q, k, spec, transformer->pos, spec->ropeTheta);
    }
    return TASK_CONTINUE;
}

int grokRmfFfn(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        transformer->rms = rms(xb2, spec->dim);
    }
    return TASK_CONTINUE;
}

int grokRmfFfnNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);

    rmsnorm(xb2, xb2, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int grokRmfFfnNormJoin(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int grokMoeRms(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
    return TASK_CONTINUE;
}

int grokMoeRmsNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsMoe, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int grokMoeRouter(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    matmul(spec->weightsFloatType, F32, block->moeRouterProbs, xb, block->moeRouter, spec->dim, spec->nExperts, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int grokMoeRouterSoftmax(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        softmax(block->moeRouterProbs, spec->nExperts);
    }
    return TASK_CONTINUE;
}

int grokMoeTopk(TASK_ARGS) {
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
    return TASK_CONTINUE;
}

int grokMoeNormWeights(TASK_ARGS) {
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
    return TASK_CONTINUE;
}

int grokQuantizeMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int grokSyncMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_INDEXES);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_WEIGHTS);
    return TASK_CONTINUE;
}

int grokMoeBlock0(TASK_ARGS) {
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
    return TASK_CONTINUE;
}

int grokMoeBlock1(TASK_ARGS) {
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
    return TASK_CONTINUE;
}

int grokQuantizeMoeMul(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_SLICED_HB, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int grokSyncMoeMulA(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int grokSyncMoeMulRearrange(TASK_ARGS) {
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
    return TASK_CONTINUE;
}

int grokSyncMoeMulB(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int grokMoeBlock2(TASK_ARGS) {
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
    return TASK_CONTINUE;
}

int grokQuantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int grokSyncMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int grokDequantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);
    return TASK_CONTINUE;
}

int grokMoeRmsFinal(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        transformer->rms = rms(xb2, spec->dim);
    }
    return TASK_CONTINUE;
}

int grokMoeRmsNormFinal(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    rmsnorm(xb2, xb2, transformer->rms, block->rmsFfn2, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int grokMoeAdd(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int grokFinalize(TASK_ARGS) {
    TASK_VARIABLES;
    if (ctx->finalize) {
        matmul(spec->weightsFloatType, F32, transformer->logits, transformer->x, transformer->wcls, spec->dim, spec->vocabSize, nThreads, threadIndex);
    }
    return TASK_CONTINUE;
}

int grokFinalize2(TASK_ARGS) {
    TASK_VARIABLES;
    if (ctx->finalize) {
        mulScalar(transformer->logits, 0.5773502691896257f, spec->vocabSize, nThreads, threadIndex);
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
    { grokMultiheadAttRope, TASK_TYPE_INFERENCE },
    { llamaMultiheadAttJoin, TASK_TYPE_INFERENCE },
    { llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE },
    { llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER },
    { llamaAtt, TASK_TYPE_INFERENCE },
    { llamaQuantizeAtt, TASK_TYPE_INFERENCE },
    { llamaSyncAtt, TASK_TYPE_TRANSFER },
    { llamaDequantizeAtt, TASK_TYPE_INFERENCE },
    { grokRmfFfn, TASK_TYPE_INFERENCE },
    { grokRmfFfnNorm, TASK_TYPE_INFERENCE },
    { grokRmfFfnNormJoin, TASK_TYPE_INFERENCE },

    { grokMoeRms, TASK_TYPE_INFERENCE },
    { grokMoeRmsNorm, TASK_TYPE_INFERENCE },
    { grokMoeRouter, TASK_TYPE_INFERENCE },
    { grokMoeRouterSoftmax, TASK_TYPE_INFERENCE },
    { grokMoeTopk, TASK_TYPE_INFERENCE },
    { grokMoeNormWeights, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeInput, TASK_TYPE_INFERENCE },
    { grokSyncMoeInput, TASK_TYPE_TRANSFER },
    { grokMoeBlock0, TASK_TYPE_INFERENCE },
    { grokMoeBlock1, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeMul, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulA, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulRearrange, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulB, TASK_TYPE_INFERENCE },
    { grokMoeBlock2, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeOutput, TASK_TYPE_INFERENCE },
    { grokSyncMoeOutput, TASK_TYPE_TRANSFER },
    { grokDequantizeMoeOutput, TASK_TYPE_INFERENCE },
    { grokMoeRmsFinal, TASK_TYPE_INFERENCE },
    { grokMoeRmsNormFinal, TASK_TYPE_INFERENCE },
    { grokMoeAdd, TASK_TYPE_INFERENCE },

    { llamaNextBlock, TASK_TYPE_INFERENCE },
    { llamaRmsFinal, TASK_TYPE_INFERENCE },
    { llamaRmsFinalNorm, TASK_TYPE_INFERENCE },
    { grokFinalize, TASK_TYPE_INFERENCE },
    { grokFinalize2, TASK_TYPE_INFERENCE },
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

    { grokSyncMoeInput, TASK_TYPE_TRANSFER },
    { grokMoeBlock0, TASK_TYPE_INFERENCE },
    { grokMoeBlock1, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeMul, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulA, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulB, TASK_TYPE_INFERENCE },
    { grokMoeBlock2, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeOutput, TASK_TYPE_INFERENCE },
    { grokSyncMoeOutput, TASK_TYPE_TRANSFER },

    { llamaNextBlock, TASK_TYPE_INFERENCE }
};

TransformerArch Grok1::arch = {
    .initInference = initInference,
    .inference = {
        .nTasks = sizeof(inferenceTasks) / sizeof(TaskLoopTask),
        .tasks = inferenceTasks
    },
    .worker = {
        .nTasks = sizeof(workerTasks) / sizeof(TaskLoopTask),
        .tasks = workerTasks
    }
};
