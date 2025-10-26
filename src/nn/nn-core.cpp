#ifdef _WIN32
    #define _USE_MATH_DEFINES
#endif
#include "nn-core.hpp"
#include <cassert>
#include <cstring>
#include <cmath>
#include <stdexcept>

// utility functions

NnSize getBytes(NnFloatType floatType, NnSize n) {
    if (floatType == F_32)
        return n * sizeof(float);
    if (floatType == F_16)
        return n * (sizeof(float) / 2);
    if (floatType == F_Q40) {
        assert(n % Q40_BLOCK_SIZE == 0);
        return (n / Q40_BLOCK_SIZE) * sizeof(NnBlockQ40);
    }
    if (floatType == F_Q80) {
        assert(n % Q80_BLOCK_SIZE == 0);
        return (n / Q80_BLOCK_SIZE) * sizeof(NnBlockQ80);
    }
    throw std::invalid_argument("Unsupported float type: " + std::to_string(floatType));
}

NnSize getBlockSize(NnFloatType floatType) {
    if (floatType == F_32)
        return 1;
    if (floatType == F_16)
        return 1;
    if (floatType == F_Q40)
        return Q40_BLOCK_SIZE;
    if (floatType == F_Q80)
        return Q80_BLOCK_SIZE;
    throw std::invalid_argument("Unsupported float type");
}

NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output) {
    // If weight=F_UNK, then returned enum should be <input>_<input>_<output>

    if (input == F_32 && output == F_32) {
        if (weight == F_UNK || weight == F_32)
            return F32_F32_F32;
        if (weight == F_Q40)
            return F32_Q40_F32;
    }
    if (input == F_32 && output == F_Q80) {
        if (weight == F_UNK || weight == F_32)
            return F32_F32_Q80;
        if (weight == F_Q40)
            return F32_Q40_Q80;
    }
    if (input == F_Q80 && output == F_32) {
        if (weight == F_UNK || weight == F_Q80)
            return Q80_Q80_F32;
        if (weight == F_32)
            return Q80_F32_F32;
        if (weight == F_Q40)
            return Q80_Q40_F32;
    }
    if (input == F_Q80 && output == F_Q80) {
        if (weight == F_UNK || weight == F_Q80)
            return Q80_Q80_Q80;
    }
    throw std::invalid_argument("Unsupported op quant: " + 
        std::string(floatTypeToString(input)) + "/" +
        std::string(floatTypeToString(weight)) + "/" +
        std::string(floatTypeToString(output)));
}

const char *opCodeToString(NnOpCode code) {
    if (code == OP_MERGE_ADD) return "MERGE_ADD";
    if (code == OP_MERGE_SUM) return "MERGE_SUM";
    if (code == OP_EMBEDDING) return "EMBEDDING";
    if (code == OP_INV_RMS) return "INV_RMS";
    if (code == OP_RMS_NORM) return "RMS_NORM";
    if (code == OP_MATMUL) return "MATMUL";
    if (code == OP_ROPE) return "ROPE";
    if (code == OP_MULTIHEAD_ATT) return "MULTIHEAD_ATT";
    if (code == OP_GELU) return "GELU";
    if (code == OP_SILU) return "SILU";
    if (code == OP_MUL) return "MUL";
    if (code == OP_SCALE) return "SCALE";
    if (code == OP_CAST) return "CAST";
    if (code == OP_REPEAT_Z) return "REPEAT_Z";
    if (code == OP_SHIFT) return "SHIFT";
    if (code == OP_SOFTMAX) return "SOFTMAX";
    if (code == OP_MOE_GATE) return "MOE_GATE";
    throw std::invalid_argument("Unknown op code: " + std::to_string(code));
}

const char *opQuantTypeToString(NnOpQuantType type) {
    if (type == F32_F32_F32) return "F32_F32_F32";
    if (type == F32_Q40_F32) return "F32_Q40_F32";
    if (type == F32_Q40_Q80) return "F32_Q40_Q80";
    if (type == F32_F32_Q80) return "F32_F32_Q80";
    if (type == Q80_Q80_Q80) return "Q80_Q80_Q80";
    if (type == Q80_Q80_F32) return "Q80_Q80_F32";
    if (type == Q80_Q40_F32) return "Q80_Q40_F32";
    if (type == Q80_F32_F32) return "Q80_F32_F32";
    throw std::invalid_argument("Unknown op quant type");
}

NnSize3D size0() {
    return { F_UNK, 0, 0, 0, 0, 0 };
}

NnSize3D size1D(NnFloatType floatType, NnUint x) {
    return size3D(floatType, 1, 1, x);
}

NnSize3D size2D(NnFloatType floatType, NnUint y, NnUint x) {
    return size3D(floatType, 1, y, x);
}

NnSize3D size3D(NnFloatType floatType, NnUint z, NnUint y, NnUint x) {
    NnSize len = z * y * x;
    NnSize lenXY = y * x;
    return { floatType, z, y, x, len, getBytes(floatType, len), getBytes(floatType, lenXY) };
}

NnPointerConfig pointerBatchConfig(NnPointerSource source, NnUint index) {
    return { source, index, PNTR_BATCH };
}

NnPointerConfig pointerBatchedSliceConfig(NnPointerSource source, NnUint index) {
    return { source, index, PNTR_BATCHED_SLICE };
}

NnPointerConfig pointerRawConfig(NnPointerSource source, NnUint index) {
    return { source, index, PNTR_RAW };
}

bool hasPointerContinuousMemory(NnPointerConfig *config) {
    if (config->type == PNTR_RAW)
        return true;
    if (config->type == PNTR_BATCH)
        return true;
    return false;
}

void releaseNetConfig(NnNetConfig *netConfig) {
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        delete[] netConfig->pipes[pipeIndex].name;
    }
    if (netConfig->nPreSyncs > 0)
        delete[] netConfig->preSyncs;
    delete[] netConfig->pipes;
}

void releaseNodeConfig(NnNodeConfig *nodeConfig) {
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segment = &nodeConfig->segments[segmentIndex];
        if (segment->nOps > 0) {
            for (NnUint opIndex = 0; opIndex < segment->nOps; opIndex++) {
                NnOpConfig *op = &segment->ops[opIndex];
                delete[] op->name;
                delete[] op->config;
            }
            delete[] segment->ops;
        }
        if (segment->nSyncs > 0)
            delete[] segment->syncs;
    }
    if (nodeConfig->nBuffers > 0) {
        for (NnUint bufferIndex = 0; bufferIndex < nodeConfig->nBuffers; bufferIndex++)
            delete[] nodeConfig->buffers[bufferIndex].name;
        delete[] nodeConfig->buffers;
    }
    delete[] nodeConfig->segments;
}

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    unsigned long total = 0;
    for (NnUint pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++)
        total += netConfig->pipes[pipeIndex].size.nBytes;
    for (NnUint bufferIndex = 0; bufferIndex < nodeConfig->nBuffers; bufferIndex++)
        total += nodeConfig->buffers[bufferIndex].size.nBytes;
    for (NnUint segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segment = &nodeConfig->segments[segmentIndex];
        for (NnUint opIndex = 0; opIndex < segment->nOps; opIndex++) {
            total += segment->ops[opIndex].weightSize.nBytes;
            total += segment->ops[opIndex].configSize;
        }
    }
    printf("📀 RequiredMemory: %lu MB\n", total / (1024 * 1024));
}

Timer::Timer() {
    reset();
}

void Timer::reset() {
    startTime = std::chrono::high_resolution_clock::now();
}

NnUint Timer::elapsedMiliseconds() {
    auto endTime = std::chrono::high_resolution_clock::now();
    return (NnUint)std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
}

NnUint Timer::elapsedMicroseconds() {
    auto endTime = std::chrono::high_resolution_clock::now();
    return (NnUint)std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

// slicers

NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes) {
    NnKvCacheSlice s;
    assert(kvDim % nNodes == 0);
    s.kvDim0 = kvDim / nNodes;
    s.keySize = size2D(F_32, seqLen, s.kvDim0);
    s.valueSize = size2D(F_32, seqLen, s.kvDim0);
    return s;
}

NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnRowMatmulSlice s;
    assert(d % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.d0 = d / nNodes;
    s.n = n;
    s.size = size2D(type, s.n, d);
    s.sliceSize = size2D(type, s.n, s.d0);
    return s;
}

NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d) {
    NnColMatmulSlice s;
    assert(n % nNodes == 0);
    s.type = type;
    s.nNodes = nNodes;
    s.n = n;
    s.n0 = n / nNodes;
    s.d = d;
    s.size = size2D(type, n, d);
    s.sliceSize = size2D(type, s.n0, d);
    return s;
}

NnRopeSlice sliceRope(NnRopeType type, NnUint qDim, NnUint kvDim, NnUint nKvHeads, NnUint nNodes, NnUint seqLen, NnUint headDim, float ropeTheta, NnUint nodeIndex) {
    NnRopeSlice s;
    assert(qDim >= kvDim);
    assert(qDim % nNodes == 0);
    assert(kvDim % nNodes == 0);

    s.kvDim = kvDim;
    s.nKvHeads = nKvHeads;
    s.seqLen = seqLen;
    s.headDim = headDim;
    s.ropeTheta = ropeTheta;

    s.qDim0 = qDim / nNodes;
    s.kvDim0 = kvDim / nNodes;
    assert(s.qDim0 % 2 == 0);
    assert(s.kvDim0 % 2 == 0);

    if (type == ROPE_LLAMA || type == ROPE_LLAMA3_1) {
        s.kvDimStart = s.kvDim0 * nodeIndex;
        s.qDimStart = s.qDim0 * nodeIndex;
        s.qDimEnd = s.qDimStart + s.qDim0;
        s.qShift = s.qDimStart - s.kvDimStart;
        s.sliceDim = s.qDimEnd - s.kvDimStart;
        assert(s.sliceDim % 2 == 0);
        s.cacheSize = size2D(F_32, seqLen, s.sliceDim);
    } else if (type == ROPE_FALCON) {
        s.cacheSize = size2D(F_32, seqLen, headDim);
    } else {
        throw std::invalid_argument("Unsupported rope type");
    }
    return s;
}

NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches) {
    NnMultiHeadAttSlice s;
    assert(nHeads % nNodes == 0);
    s.nHeads = nHeads;
    s.nHeads0 = nHeads / nNodes;
    s.attSize = size2D(F_32, nBatches, s.nHeads0 * seqLen);
    return s;
}

// splitters

NnUint splitRowMatmulWeight(NnRowMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);
    assert(slice->n % blockSize == 0);

    NnSize n = slice->n / blockSize;
    NnSize offset = slice->d0 * nodeIndex * n * batchBytes;
    NnSize copiedBytes = 0;
    for (NnUint d = 0; d < slice->d0; d++) {
        for (NnUint j = 0; j < n; j++) {
            NnSize o = (d * n + j) * batchBytes;
            std::memcpy(weight0 + o, weight + offset + o, batchBytes);
            copiedBytes += batchBytes;
        }
    }
    return copiedBytes;
}

NnUint splitColMatmulWeight(NnColMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);
    assert(slice->n0 % blockSize == 0);

    NnSize n = slice->n / blockSize;
    NnSize rowBytes = n * batchBytes;
    NnSize row0Bytes = (slice->n0 / blockSize) * batchBytes;
    NnSize rowOffsetBytes = nodeIndex * row0Bytes;
    NnSize copiedBytes = 0;
    for (NnUint d = 0; d < slice->d; d++) {
        std::memcpy(&weight0[row0Bytes * d], &weight[rowBytes * d + rowOffsetBytes], row0Bytes);
        copiedBytes += row0Bytes;
    }
    return copiedBytes;
}

// helper

static inline float scaleFrequencyLlama3(const float freq, const NnRopeOpConfig *config) {
    // https://github.com/meta-llama/llama-models/blob/4269717b2ea587627903bacbb75ccce1427ad914/models/llama3/reference_impl/model.py#L55
    const float waveLen = 2.0f * M_PI / freq;
    const float highFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingHighFreqFactor;
    if (waveLen < highFreqWavelen) {
        return freq;
    }
    const float lowFreqWavelen = config->ropeScalingOrigMaxSeqLen / config->ropeScalingLowFreqFactor;
    if (waveLen > lowFreqWavelen) {
        return freq / config->ropeScalingFactor;
    }
    const float smooth = (config->ropeScalingOrigMaxSeqLen / waveLen - config->ropeScalingLowFreqFactor) /
        (config->ropeScalingHighFreqFactor - config->ropeScalingLowFreqFactor);
    return (1 - smooth) * freq / config->ropeScalingFactor + smooth * freq;
}

static inline void fullfillRopeLlamaCache(const NnRopeOpConfig *config, float *cache) {
    assert((config->slice.qDimEnd - config->slice.kvDimStart) % 2 == 0);

    const bool applyScaling = config->ropeScalingFactor != 1.0f;
    for (NnUint pos = 0; pos < config->slice.seqLen; pos++) {
        for (NnUint i = config->slice.kvDimStart; i < config->slice.qDimEnd; i += 2) {
            const NnUint h = i % config->slice.headDim;
            float freq = 1.0f / powf(config->slice.ropeTheta, h / (float)config->slice.headDim);
            if (applyScaling)
                freq = scaleFrequencyLlama3(freq, config);
            const float val = pos * freq;
            const float fcr = cosf(val);
            const float fci = sinf(val);
            cache[pos * config->slice.sliceDim + (i - config->slice.kvDimStart)] = fcr;
            cache[pos * config->slice.sliceDim + (i - config->slice.kvDimStart) + 1] = fci;
        }
    }
}

static inline void fullfillRopeFalconCache(const NnRopeOpConfig *config, float *cache) {
    const float hs = (float)config->slice.headDim;

    for (NnUint pos = 0; pos < config->slice.seqLen; pos++) {
        for (NnUint j = 0; j < config->slice.headDim / 2; j++) {
            const float freq = 1.0f / powf(config->slice.ropeTheta, 2.0f * (float)(j / hs));
            const float val = pos * freq;
            const float fcr = cosf(val);
            const float fci = sinf(val);
            cache[pos * config->slice.headDim + j] = fcr;
            cache[pos * config->slice.headDim + j + config->slice.headDim / 2] = fci;
        }
    }
}

void fullfillRopeCache(const NnRopeOpConfig *config, float *cache) {
    if (config->type == ROPE_LLAMA || config->type == ROPE_LLAMA3_1)
        fullfillRopeLlamaCache(config, cache);
    else if (config->type == ROPE_FALCON)
        fullfillRopeFalconCache(config, cache);
    else
        throw std::invalid_argument("Unsupported rope type");
}
