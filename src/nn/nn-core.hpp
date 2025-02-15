#ifndef NN_CORE_H
#define NN_CORE_H

#include <chrono>
#include <list>
#include <memory>
#include <cstdint>
#include "nn-quants.hpp"

// primitives

typedef struct {
    NnFloatType floatType;
    NnSize y;
    NnSize x;
    NnSize length;
    NnSize nBytes;
} NnSize2D;

// slices

typedef struct {
    NnSize kvDim0;
    NnSize2D keySize;
    NnSize2D valueSize;
} NnKvCacheSlice;

typedef struct {
    NnFloatType type;
    NnSize nNodes;
    NnSize d0;
    NnSize n;
    NnSize2D size;
    NnSize2D sliceSize;
} NnRowMatmulSlice;

typedef struct {
    NnFloatType type;
    NnSize nNodes;
    NnSize n;
    NnSize n0;
    NnSize d;
    NnSize2D size;
    NnSize2D sliceSize;
} NnColMatmulSlice;

typedef struct {
    NnSize qDim0;
    NnSize qDimStart;
    NnSize qDimEnd;
    NnSize qShift;
    NnSize kvDim;
    NnSize kvDim0;
    NnSize kvDimStart;
    NnSize sliceDim;
    NnSize seqLen;
    NnSize headSize;
    NnSize nKvHeads;
    float ropeTheta;
    NnSize2D cacheSize;
} NnRopeSlice;

typedef struct {
    NnSize nHeads;
    NnSize nHeads0;
    NnSize2D attSize;
} NnMultiHeadAttSlice;

// base enums

enum NnOpCode {
    OP_MERGE_ADD,
    OP_EMBEDDING,
    OP_INV_RMS,
    OP_RMS_NORM,
    OP_MATMUL,
    OP_ROPE_LLAMA,
    OP_MULTIHEAD_ATT,
    OP_GELU,
    OP_SILU,
    OP_MUL,
    OP_CAST,
};

enum NnOpQuantType {
    // <input>_<weight>_<output>
    F32_F32_F32,
    F32_Q40_F32,
    F32_Q40_Q80,
    F32_F32_Q80,
    Q80_Q80_Q80,
    Q80_Q80_F32,
    Q80_Q40_F32,
    Q80_F32_F32,
};

#define N_OP_CODES (OP_CAST + 1)
#define N_OP_QUANTS (Q80_F32_F32 + 1)

enum NnPointerType {
    PNTR_PIPE,
    PNTR_BUFFER,
};

enum NnPointerSliceType {
    SLICE_NONE,
    SLICE_NODE_PART,
};

enum NnPointerBatchType {
    PNTR_BATCH_DEFAULT,
    PNTR_BATCH_PIPE,
};

enum NnSyncType {
    SYNC_WITH_ROOT, // whole pipe to all nodes
    SYNC_NODE_SLICES, // my slice of pipe to all nodes
    SYNC_NODE_SLICES_EXCEPT_ROOT, // only workers send slices to root, root does not send
};

enum NnRopeType {
    ROPE_LLAMA = 0,
    ROPE_FALCON = 1,
    ROPE_LLAMA3_1 = 2,
};

// base configs

typedef struct {
    char *name;
    NnSize2D size;
} NnPipeConfig;

typedef struct {
    char *name;
    NnSize2D size;
} NnBufferConfig;

typedef struct {
    NnPointerType pointerType;
    NnSize pointerIndex;
    NnPointerSliceType sliceType;
    NnPointerBatchType batchType;
    NnSize batchArg0;
} NnPointerConfig;

typedef struct {
    NnOpCode code;
    char *name;
    NnSize index;
    NnPointerConfig input;
    NnPointerConfig output;
    NnSize2D weightSize;
    NnByte *config;
    NnSize configSize;
} NnOpConfig;

typedef struct {
    NnSize pipeIndex;
    NnSyncType syncType;
} NnSyncConfig;

typedef struct  {
    NnSize nOps;
    NnOpConfig *ops;
    NnSize nSyncs;
    NnSyncConfig *syncs;
    bool syncPointers;
} NnSegmentConfig;

typedef struct {
    NnSize nBatches;
    NnSize nNodes;
    NnSize nPipes;
    NnPipeConfig *pipes;
} NnNetConfig;

typedef struct {
    NnSize nodeIndex;
    NnSize nBuffers;
    NnBufferConfig *buffers;
    NnSize nSegments;
    NnSegmentConfig *segments;
} NnNodeConfig;

// op configs

typedef struct {
    // empty
} NnEmbeddingOpConfig;

typedef struct {
    float epsilon;
} NnInvRmsOpConfig;

typedef struct {
    NnSize invRmsBufferIndex;
} NnRmsNormOpConfig;

typedef struct {
    // empty
} NnMatmulOpConfig;

typedef struct {
    bool isQ;
    NnSize positionPipeIndex;
    NnSize ropeCacheBufferIndex;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnSize ropeScalingOrigMaxSeqLen;
    NnRopeSlice slice;
} NnRopeLlamaOpConfig;

typedef struct {
    NnSize nKvHeads;
    NnSize headSize;
    NnSize seqLen;
    NnSize positionPipeIndex;
    NnSize queryBufferIndex;
    NnSize keyCacheBufferIndex;
    NnSize valueCacheBufferIndex;
    NnSize attBufferIndex;
    NnRowMatmulSlice qSlice;
    NnKvCacheSlice kvCacheSlice;
    NnMultiHeadAttSlice multiHeadAttSlice;
} NnMultiHeadAttOpConfig;

typedef struct {
    // empty
} NnMergeAddOpCodeConfig;

typedef struct {
    // empty
} NnSiluOpCodeConfig;

typedef struct {
    // empty
} NMulOpCodeConfig;

typedef struct {
    // empty
} NnCastOpCodeConfig;

// utility functions

const char *opCodeToString(NnOpCode code);
const char *opQuantTypeToString(NnOpQuantType type);

NnSize getBytes(NnFloatType floatType, NnSize n);
NnSize getBlockSize(NnFloatType floatType);
NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output);
NnSize2D size0();
NnSize2D size1D(NnFloatType floatType, NnSize x);
NnSize2D size2D(NnFloatType floatType, NnSize y, NnSize x);
NnPointerConfig pointerConfig(NnPointerType type, NnSize index);
NnPointerConfig pointerConfigWithPipedBatch(NnPointerType type, NnSize index, NnSize pipeIndex);
NnPointerConfig slicedPointerConfig(NnPointerType type, NnSize index);
bool hasPointerContinuousMemory(NnPointerConfig *config);

void releaseNetConfig(NnNetConfig *netConfig);
void releaseNodeConfig(NnNodeConfig *nodeConfig);

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig);

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    Timer();
    NnSize elapsedMiliseconds();
    NnSize elapsedMicroseconds();
};

// slicers

NnKvCacheSlice sliceKvCache(NnSize kvDim, NnSize seqLen, NnSize nNodes);
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnSize nNodes, NnSize n, NnSize d);
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnSize nNodes, NnSize n, NnSize d);
NnRopeSlice sliceRope(NnSize dim, NnSize kvDim, NnSize nKvHeads, NnSize nNodes, NnSize seqLen, NnSize headSize, float ropeTheta, NnSize nodeIndex);
NnMultiHeadAttSlice sliceMultiHeadAtt(NnSize nHeads, NnSize seqLen, NnSize nNodes);

// splitters

NnSize splitRowMatmulWeight(NnRowMatmulSlice *slice, NnSize nodeIndex, NnByte *weight, NnByte *weight0);
NnSize splitColMatmulWeight(NnColMatmulSlice *slice, NnSize nodeIndex, NnByte *weight, NnByte *weight0);

#endif
