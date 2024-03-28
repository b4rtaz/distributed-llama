#include <cmath>
#include <cassert>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "transformer-tasks.hpp"

#define TASK_ARGS unsigned int nThreads, unsigned int threadIndex, void* userData

#define TASK_VARIABLES \
    TransformerContext* ctx = (TransformerContext*)userData; \
    Transformer* transformer = ctx->transformer; \
    TransformerBlock* block = transformer->blocks[ctx->currentBlockIndex]; \
    TransformerSpec* spec = transformer->spec;

void syncUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    char* buffer = ctx->transformer->buffer->getUnit(bufferIndex);
    size_t bufferBytes = ctx->transformer->buffer->getUnitBytes(bufferIndex);

    if (ctx->socketPool != NULL) {
        // root

        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        SocketIo ios[nSockets];
        for (int i = 0; i < nSockets; i++) {
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = buffer;
            ios[i].size = bufferBytes;
        }
        ctx->socketPool->writeMany(nSockets, ios);
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return;

        // worker
        ctx->socket->read(buffer, bufferBytes);
    }
}

void syncSliceOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    size_t bufferBytes = ctx->transformer->buffer->getSlicedBytes(bufferIndex);
    if (ctx->socketPool != NULL) {
        // root

        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        SocketIo ios[nSockets];
        for (int i = 0; i < nSockets; i++) {
            int socketIndex = threadIndex + i * nThreads;
            uint8_t workerSliceIndex = socketIndex + 1;
            ios[i].socketIndex = socketIndex;
            ios[i].data = ctx->transformer->buffer->getSliced(bufferIndex, workerSliceIndex);
            ios[i].size = bufferBytes;
        }

        ctx->socketPool->readMany(nSockets, ios);
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return;

        // worker
        char* buffer = ctx->transformer->buffer->getSliced(bufferIndex, ctx->transformer->sliceIndex);
        ctx->socket->write(buffer, bufferBytes);
    }
}

void syncMissingSlicesOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    size_t sliceBytes = ctx->transformer->buffer->getSlicedBytes(bufferIndex);
    if (ctx->socketPool != NULL) {
        // root

        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        SocketIo ios[nSockets];

        for (uint8_t si = 0; si < ctx->transformer->spec->nSlices - 1; si++) {
            for (unsigned int i = 0; i < nSockets; i++) {
                int socketIndex = threadIndex + i * nThreads;
                uint8_t workerSliceIndex = socketIndex + 1;
                uint8_t sliceIndex = si < workerSliceIndex ? si : si + 1;
                ios[i].socketIndex = socketIndex;
                ios[i].data = ctx->transformer->buffer->getSliced(bufferIndex, sliceIndex);
                ios[i].size = sliceBytes;
            }
            ctx->socketPool->writeMany(nSockets, ios);
        }
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return;

        // worker
        for (uint8_t sliceIndex = 0; sliceIndex < ctx->transformer->spec->nSlices; sliceIndex++) {
            if (sliceIndex != ctx->transformer->sliceIndex) {
                char* buffer = ctx->transformer->buffer->getSliced(bufferIndex, sliceIndex);
                ctx->socket->read(buffer, sliceBytes);
            }
        }
    }
}

void quantizeUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);

    quantizeQ80Row(
        (float*)ctx->transformer->buffer->getUnit(sourceBufferIndex),
        (BlockQ80*)ctx->transformer->buffer->getUnit(targetBufferIndex),
        ctx->transformer->buffer->getUnitBytes(sourceBufferIndex) / sizeof(float),
        nThreads,
        threadIndex);
}

void quantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool quantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    if (ctx->transformer->sliceIndex == 0 && !quantizeRootSlice) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);

    quantizeQ80Row(
        (float*)ctx->transformer->buffer->getSliced(sourceBufferIndex, ctx->transformer->sliceIndex),
        (BlockQ80*)ctx->transformer->buffer->getSliced(targetBufferIndex, ctx->transformer->sliceIndex),
        ctx->transformer->buffer->getSlicedBytes(sourceBufferIndex) / sizeof(float),
        nThreads,
        threadIndex);
}

void dequantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool dequantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);
    assert(ctx->socketPool != NULL); // This function may be called only by root.

    unsigned int sliceIndex = dequantizeRootSlice ? 0 : 1;
    for (; sliceIndex < ctx->transformer->spec->nSlices; sliceIndex++) {
        dequantizeQ80Row(
            (BlockQ80*)ctx->transformer->buffer->getSliced(sourceBufferIndex, sliceIndex),
            (float*)ctx->transformer->buffer->getSliced(targetBufferIndex, sliceIndex),
            (ctx->transformer->buffer->getSlicedBytes(sourceBufferIndex) / sizeof(BlockQ80)) * QK80,
            nThreads,
            threadIndex);
    }
}

//

int rmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
    return TASK_CONTINUE;
}

int rmsAttNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsAtt, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int quantizeRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int syncRmsAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int qkv(TASK_ARGS) {
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

int quantizeQkv(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_Q, TB_SLICED_Q_QUANTIZED);
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_K, TB_SLICED_K_QUANTIZED);
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_V, TB_SLICED_V_QUANTIZED);
    return TASK_CONTINUE;
}

int syncQkv(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_Q_QUANTIZED);
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_K_QUANTIZED);
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_V_QUANTIZED);
    // if (ctx->socketPool != NULL && threadIndex == 0) { float* v = (float*)block->q0; printf("q0 (%d): %f %f %f %f %f %f\n", ctx->currentBlockIndex, v[0], v[1], v[2], v[3], v[4], v[5]); }
    return TASK_CONTINUE;
}

int dequantizeQkv(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_Q_QUANTIZED, TB_SLICED_Q);
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_K_QUANTIZED, TB_SLICED_K);
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_V_QUANTIZED, TB_SLICED_V);
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

int quantizeMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int syncMultiheadAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int att(TASK_ARGS) {
    TASK_VARIABLES;

    char* xb = transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* xb2 = (float*)transformer->buffer->getSliced(TB_SLICED_XB2, transformer->sliceIndex);

    matmul(spec->weightsFloatType, spec->bufferFloatType, xb2, xb, block->wo0, block->wo0Slice->n, block->wo0Slice->d0, nThreads, threadIndex);

    return TASK_CONTINUE;
}

int quantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int syncAtt(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int dequantizeAtt(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);    
    return TASK_CONTINUE;
}

int rmfFfn(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0) {
        float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
        transformer->rms = rms(xb2, spec->dim);
    }
    return TASK_CONTINUE;
}

int rmfFfnNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);

    rmsnorm(xb2, xb2, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int rmfFfnNormJoin(TASK_ARGS) {
    TASK_VARIABLES;

    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(transformer->x, xb2, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int moeRms(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
    return TASK_CONTINUE;
}

int moeRmsNorm(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsMoe, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int moeRouter(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    matmul(spec->weightsFloatType, F32, block->moeRouterProbs, xb, block->moeRouter, spec->dim, spec->nExperts, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int moeRouterSoftmax(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        softmax(block->moeRouterProbs, spec->nExperts);
    }
    return TASK_CONTINUE;
}

int moeTopk(TASK_ARGS) {
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

int moeNormWeights(TASK_ARGS) {
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

int quantizeMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
    return TASK_CONTINUE;
}

int syncMoeInput(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_INDEXES);
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_MOE_WEIGHTS);
    return TASK_CONTINUE;
}

int moeBlock0(TASK_ARGS) {
    TASK_VARIABLES;

    uint8_t* indexes = (uint8_t*)transformer->buffer->getUnit(TB_UNIT_MOE_INDEXES);
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);
    float* hb = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        uint8_t e = indexes[ae];

        float* expertUp = &hb[block->moeUp0Slice->d0 * ae];
        float* expertGate = &block->expertGate[block->moeGate0Slice->d0 * ae];
        matmul(spec->weightsFloatType, spec->bufferFloatType, expertUp, xb, block->moeUp[e], block->moeUp0Slice->n, block->moeUp0Slice->d0, nThreads, threadIndex);
        matmul(spec->weightsFloatType, spec->bufferFloatType, expertGate, xb, block->moeGate[e], block->moeGate0Slice->n, block->moeGate0Slice->d0, nThreads, threadIndex);
    }
    return TASK_CONTINUE;
}

int moeBlock1(TASK_ARGS) {
    TASK_VARIABLES;
    float* hb = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex);

    for (int ae = 0; ae < spec->nActiveExperts; ae++) {
        float* expertUp = &hb[block->moeUp0Slice->d0 * ae];
        float* expertGate = &block->expertGate[block->moeGate0Slice->d0 * ae];

        gelu(expertGate, block->moeGate0Slice->d0, nThreads, threadIndex);
        mul(expertUp, expertGate, block->moeGate0Slice->d0, nThreads, threadIndex);
    }
    return TASK_CONTINUE;
}

int quantizeMoeMul(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_SLICED_HB, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int syncMoeMulA(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int syncMoeMulRearrange(TASK_ARGS) {
    TASK_VARIABLES;

    if (threadIndex == 0 && spec->nSlices > 1) {
        size_t rowBytes = getBatchBytes(spec->bufferFloatType, spec->hiddenDim, 1);
        size_t dBytes = getBatchBytes(spec->bufferFloatType, block->moeUp0Slice->d0, 1);

        char* hbq = transformer->buffer->getUnit(TB_SLICED_HB_QUANTIZED);
        size_t bufferBytes = transformer->buffer->getUnitBytes(TB_SLICED_HB_QUANTIZED);
        char* buffer = new char[bufferBytes];

        for (int s = 0; s < spec->nSlices; s++) {
            for (int ae = 0; ae < spec->nActiveExperts; ae++) {
                memcpy(&buffer[ae * rowBytes + s * dBytes], &hbq[s * rowBytes + ae * dBytes], dBytes);
            }
        }

        memcpy(hbq, buffer, rowBytes * spec->nActiveExperts);
        delete[] buffer;
    }
    return TASK_CONTINUE;
}

int syncMoeMulB(TASK_ARGS) {
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_SLICED_HB_QUANTIZED);
    return TASK_CONTINUE;
}

int moeBlock2(TASK_ARGS) {
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

int quantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int syncMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XB2_QUANTIZED);
    return TASK_CONTINUE;
}

int dequantizeMoeOutput(TASK_ARGS) {
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XB2_QUANTIZED, TB_SLICED_XB2);
    return TASK_CONTINUE;
}

int moeJoin(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* xb2 = (float*)transformer->buffer->getUnit(TB_SLICED_XB2);
    add(xb, xb2, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int moeRmsFinal(TASK_ARGS) {
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
        transformer->rms = rms(xb, spec->dim);
    }
    return TASK_CONTINUE;
}

int moeRmsNormFinal(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, xb, transformer->rms, block->rmsFfn2, spec->dim, nThreads, threadIndex);
    return TASK_CONTINUE;
}

int moeAdd(TASK_ARGS) {
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    add(transformer->x, xb, spec->dim, nThreads, threadIndex);
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
        matmul(spec->weightsFloatType, F32, transformer->logits, x, transformer->wcls, spec->dim, spec->vocabSize, nThreads, threadIndex);
    }
    return TASK_CONTINUE;
}

int finalize2(TASK_ARGS) {
    TASK_VARIABLES;
    if (ctx->finalize) {
        mulScalar(transformer->logits, 0.5773502691896257f, spec->vocabSize, nThreads, threadIndex);
        return TASK_STOP;
    }
    return TASK_CONTINUE;
}

static TaskLoopTask inferenceTasks[] = {
    { rmsAtt, TASK_TYPE_INFERENCE },
    { rmsAttNorm, TASK_TYPE_INFERENCE },
    { quantizeRmsAtt, TASK_TYPE_INFERENCE },
    { syncRmsAtt, TASK_TYPE_TRANSFER },
    { qkv, TASK_TYPE_INFERENCE },
    { quantizeQkv, TASK_TYPE_INFERENCE },
    { syncQkv, TASK_TYPE_TRANSFER },
    { dequantizeQkv, TASK_TYPE_INFERENCE },
    { multiheadAtt, TASK_TYPE_INFERENCE },
    { quantizeMultiheadAtt, TASK_TYPE_INFERENCE },
    { syncMultiheadAtt, TASK_TYPE_TRANSFER },
    { att, TASK_TYPE_INFERENCE },
    { quantizeAtt, TASK_TYPE_INFERENCE },
    { syncAtt, TASK_TYPE_TRANSFER },
    { dequantizeAtt, TASK_TYPE_INFERENCE },
    { rmfFfn, TASK_TYPE_INFERENCE },
    { rmfFfnNorm, TASK_TYPE_INFERENCE },
    { rmfFfnNormJoin, TASK_TYPE_INFERENCE },

    { moeRms, TASK_TYPE_INFERENCE },
    { moeRmsNorm, TASK_TYPE_INFERENCE },
    { moeRouter, TASK_TYPE_INFERENCE },
    { moeRouterSoftmax, TASK_TYPE_INFERENCE },
    { moeTopk, TASK_TYPE_INFERENCE },
    { moeNormWeights, TASK_TYPE_INFERENCE },
    { quantizeMoeInput, TASK_TYPE_INFERENCE },
    { syncMoeInput, TASK_TYPE_TRANSFER },
    { moeBlock0, TASK_TYPE_INFERENCE },
    { moeBlock1, TASK_TYPE_INFERENCE },
    { quantizeMoeMul, TASK_TYPE_INFERENCE },
    { syncMoeMulA, TASK_TYPE_INFERENCE },
    { syncMoeMulRearrange, TASK_TYPE_INFERENCE },
    { syncMoeMulB, TASK_TYPE_INFERENCE },
    { moeBlock2, TASK_TYPE_INFERENCE },
    { quantizeMoeOutput, TASK_TYPE_INFERENCE },
    { syncMoeOutput, TASK_TYPE_TRANSFER },
    { dequantizeMoeOutput, TASK_TYPE_INFERENCE },
    { moeJoin, TASK_TYPE_INFERENCE },
    { moeRmsFinal, TASK_TYPE_INFERENCE },
    { moeRmsNormFinal, TASK_TYPE_INFERENCE },
    { moeAdd, TASK_TYPE_INFERENCE },

    { nextBlock, TASK_TYPE_INFERENCE },
    { rmsFinal, TASK_TYPE_INFERENCE },
    { rmsFinalNorm, TASK_TYPE_INFERENCE },
    { finalize, TASK_TYPE_INFERENCE },
    { finalize2, TASK_TYPE_INFERENCE },
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

    // grok
    for (int i = 0; i < transformer->spec->dim; i++) {
        transformer->x[i] *= 78.38367176906169f;
    }

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
    { quantizeQkv, TASK_TYPE_INFERENCE },
    { syncQkv, TASK_TYPE_TRANSFER },
    { syncMultiheadAtt, TASK_TYPE_TRANSFER },
    { att, TASK_TYPE_INFERENCE },
    { quantizeAtt, TASK_TYPE_INFERENCE },
    { syncAtt, TASK_TYPE_TRANSFER },

    { syncMoeInput, TASK_TYPE_TRANSFER },
    { moeBlock0, TASK_TYPE_INFERENCE },
    { moeBlock1, TASK_TYPE_INFERENCE },
    { quantizeMoeMul, TASK_TYPE_INFERENCE },
    { syncMoeMulA, TASK_TYPE_INFERENCE },
    { syncMoeMulB, TASK_TYPE_INFERENCE },
    { moeBlock2, TASK_TYPE_INFERENCE },
    { quantizeMoeOutput, TASK_TYPE_INFERENCE },
    { syncMoeOutput, TASK_TYPE_TRANSFER },

    { nextBlock, TASK_TYPE_INFERENCE }
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
