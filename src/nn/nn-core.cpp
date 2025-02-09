#include "nn-core.hpp"
#include <cassert>
#include <cstring>
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
    if (code == OP_EMBEDDING) return "EMBEDDING";
    if (code == OP_INV_RMS) return "INV_RMS";
    if (code == OP_RMS_NORM) return "RMS_NORM";
    if (code == OP_MATMUL) return "MATMUL";
    if (code == OP_ROPE_LLAMA) return "ROPE_LLAMA";
    if (code == OP_MULTIHEAD_ATT) return "MULTIHEAD_ATT";
    if (code == OP_GELU) return "GELU";
    if (code == OP_SILU) return "SILU";
    if (code == OP_MUL) return "MUL";
    if (code == OP_CAST) return "CAST";
    throw std::invalid_argument("Unknown op code");
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

NnSize2D size0() {
    return { F_UNK, 0, 0, 0, 0 };
}

NnSize2D size1D(NnFloatType floatType, NnSize x) {
    return size2D(floatType, 1, x);
}

NnSize2D size2D(NnFloatType floatType, NnSize y, NnSize x) {
    NnSize length = y * x;
    return { floatType, y, x, length, getBytes(floatType, length) };
}

NnPointerConfig pointerConfig(NnPointerType type, NnSize index) {
    return { type, index, SLICE_NONE, PNTR_BATCH_DEFAULT, 0 /* not used*/ };
}

NnPointerConfig pointerConfigWithPipedBatch(NnPointerType type, NnSize index, NnSize pipeIndex) {
    return { type, index, SLICE_NONE, PNTR_BATCH_PIPE, pipeIndex };
}

NnPointerConfig slicedPointerConfig(NnPointerType type, NnSize index) {
    return { type, index, SLICE_NODE_PART, PNTR_BATCH_DEFAULT, 0 /* not used*/ };
}

bool hasPointerContinuousMemory(NnPointerConfig *config) {
    return config->batchType == PNTR_BATCH_DEFAULT && config->sliceType == SLICE_NONE;
}

void releaseNetConfig(NnNetConfig *netConfig) {
    for (NnSize pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        delete[] netConfig->pipes[pipeIndex].name;
    }
    delete[] netConfig->pipes;
}

void releaseNodeConfig(NnNodeConfig *nodeConfig) {
    for (NnSize segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segment = &nodeConfig->segments[segmentIndex];
        if (segment->nOps > 0) {
            for (NnSize opIndex = 0; opIndex < segment->nOps; opIndex++) {
                NnOpConfig *op = &segment->ops[opIndex];
                delete[] op->name;
                delete[] op->config;
            }
            delete[] segment->ops;
        }
        if (segment->nSyncs > 0)
            delete[] segment->syncs;
    }
    for (NnSize bufferIndex = 0; bufferIndex < nodeConfig->nBuffers; bufferIndex++)
        delete[] nodeConfig->buffers[bufferIndex].name;
    delete[] nodeConfig->buffers;
    delete[] nodeConfig->segments;
}

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    unsigned long total = 0;
    for (NnSize pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++)
        total += netConfig->pipes[pipeIndex].size.nBytes;
    for (NnSize bufferIndex = 0; bufferIndex < nodeConfig->nBuffers; bufferIndex++)
        total += nodeConfig->buffers[bufferIndex].size.nBytes;
    for (NnSize segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segment = &nodeConfig->segments[segmentIndex];
        for (NnSize opIndex = 0; opIndex < segment->nOps; opIndex++) {
            total += segment->ops[opIndex].weightSize.nBytes;
            total += segment->ops[opIndex].configSize;
        }
    }
    printf("📀 RequiredMemory: %lu kB\n", total / 1024);
}

// slicers

NnKvCacheSlice sliceKvCache(NnSize kvDim, NnSize seqLen, NnSize nNodes) {
    NnKvCacheSlice s;
    assert(kvDim % nNodes == 0);
    s.kvDim0 = kvDim / nNodes;
    s.keySize = size2D(F_32, seqLen, s.kvDim0);
    s.valueSize = size2D(F_32, seqLen, s.kvDim0);
    return s;
}

NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnSize nNodes, NnSize n, NnSize d) {
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

NnColMatmulSlice sliceColMatmul(NnFloatType type, NnSize nNodes, NnSize n, NnSize d) {
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

NnRopeSlice sliceRope(NnSize dim, NnSize kvDim, NnSize nKvHeads, NnSize nNodes, NnSize seqLen, NnSize headSize, float ropeTheta, NnSize nodeIndex) {
    NnRopeSlice s;
    assert(dim >= kvDim);
    assert(dim % nNodes == 0);
    assert(kvDim % nNodes == 0);

    s.qDim0 = dim / nNodes;
    s.kvDim0 = kvDim / nNodes;
    assert(s.qDim0 % 2 == 0);
    assert(s.kvDim0 % 2 == 0);

    s.kvDimStart = s.kvDim0 * nodeIndex;
    s.qDimStart = s.qDim0 * nodeIndex;
    s.qDimEnd = s.qDimStart + s.qDim0;
    s.qShift = s.qDimStart - s.kvDimStart;
    s.sliceDim = s.qDimEnd - s.kvDimStart;
    assert(s.sliceDim % 2 == 0);

    s.kvDim = kvDim;
    s.nKvHeads = nKvHeads;
    s.seqLen = seqLen;
    s.headSize = headSize;
    s.ropeTheta = ropeTheta;
    s.cacheSize = size2D(F_32, s.seqLen, s.sliceDim);
    return s;
}

NnMultiHeadAttSlice sliceMultiHeadAtt(NnSize nHeads, NnSize seqLen, NnSize nNodes) {
    NnMultiHeadAttSlice s;
    assert(nHeads % nNodes == 0);
    s.nHeads = nHeads;
    s.nHeads0 = nHeads / nNodes;
    s.attSize = size2D(F_32, seqLen, s.nHeads0);
    return s;
}

// splitters

NnSize splitRowMatmulWeight(NnRowMatmulSlice *slice, NnSize nodeIndex, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);
    assert(slice->n % blockSize == 0);

    NnSize n = slice->n / blockSize;
    NnSize offset = slice->d0 * nodeIndex * n * batchBytes;
    NnSize copiedBytes = 0;
    for (NnSize d = 0; d < slice->d0; d++) {
        for (NnSize j = 0; j < n; j++) {
            NnSize o = (d * n + j) * batchBytes;
            std::memcpy(weight0 + o, weight + offset + o, batchBytes);
            copiedBytes += batchBytes;
        }
    }
    return copiedBytes;
}

NnSize splitColMatmulWeight(NnColMatmulSlice *slice, NnSize nodeIndex, NnByte *weight, NnByte *weight0) {
    NnSize blockSize = getBlockSize(slice->type);
    NnSize batchBytes = getBytes(slice->type, blockSize);
    assert(slice->n0 % blockSize == 0);

    NnSize n = slice->n / blockSize;
    NnSize rowBytes = n * batchBytes;
    NnSize row0Bytes = (slice->n0 / blockSize) * batchBytes;
    NnSize rowOffsetBytes = nodeIndex * row0Bytes;
    NnSize copiedBytes = 0;
    for (NnSize d = 0; d < slice->d; d++) {
        std::memcpy(&weight0[row0Bytes * d], &weight[rowBytes * d + rowOffsetBytes], row0Bytes);
        copiedBytes += row0Bytes;
    }
    return copiedBytes;
}
