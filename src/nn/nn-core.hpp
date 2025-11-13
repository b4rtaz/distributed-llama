#ifndef NN_CORE_H
#define NN_CORE_H

#include <chrono>
#include <list>
#include <memory>
#include <cstdint>
#include "nn-quants.hpp"
#include <vector>

// primitives

typedef struct {
    NnFloatType floatType;
    NnUint z;
    NnUint y;
    NnUint x;
    NnSize length;
    NnSize nBytes;
    NnSize nBytesXY;
} NnSize3D;

// slices

typedef struct {
    NnUint kvDim0;
    NnSize3D keySize;
    NnSize3D valueSize;
} NnKvCacheSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint d0;
    NnUint n;
    NnSize3D size;
    NnSize3D sliceSize;
} NnRowMatmulSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint n;
    NnUint n0;
    NnUint d;
    NnSize3D size;
    NnSize3D sliceSize;
} NnColMatmulSlice;

typedef struct {
    NnUint qDim0;
    NnUint qDimStart;
    NnUint qDimEnd;
    NnUint qShift;
    NnUint kvDim;
    NnUint kvDim0;
    NnUint kvDimStart;
    NnUint sliceDim;
    NnUint seqLen;
    NnUint headDim;
    NnUint nKvHeads;
    float ropeTheta;
    NnSize3D cacheSize;
} NnRopeSlice;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnSize3D attSize;
} NnMultiHeadAttSlice;

//Uneven slice
typedef struct {
    NnUint kvStart;   // 在 kv 维上的起点（本节点）
    NnUint kvLen;     // 在 kv 维上的长度（本节点）
 
    NnUint kvDim0;    // 本节点的 kvDim 局部（可与 kvLen 对齐；保留以兼容旧逻辑）
    NnSize3D keySize;    // 本节点局部 K 缓冲尺寸
    NnSize3D valueSize;  // 本节点局部 V 缓冲尺寸
} NnKvCacheSliceUneven;

typedef struct {
    NnFloatType type;


    NnUint inStart;   // 在输入维 D_in 上的起点（本节点）
    NnUint inLen;     // 在输入维 D_in 上的长度（本节点）

    // —— 语义收敛为“本节点局部” —— 
    // 原来用于均匀切的 nNodes 建议删除；若保留请忽略
    // NnUint nNodes;  // <- 不再使用，或在构图时仅作信息提示

    // d0：本节点 matmul 结果的列数（对 Wq/Wk/Wv，等于本节点参与乘积后的输出列）
    NnUint d0;
    NnUint n;
    NnSize3D size;

    // 本节点局部权重切片的 size（保持原字段名，但明确“局部化”）
    NnSize3D sliceSize;
} NnRowMatmulSliceUneven; 

typedef struct {
    NnFloatType type;

    // —— 新增：非均匀所需（本节点负责的“被切维度”区间）——
    NnUint outStart;  // 在输出维 D_out 上的起点（本节点）
    NnUint outLen;    // 在输出维 D_out 上的长度（本节点）

    // —— 语义收敛为“本节点局部” —— 
    // NnUint nNodes;  // <- 不再使用，或忽略
    NnUint n;         
    NnUint n0;        
    NnUint d;         
    NnSize3D size;        
    NnSize3D sliceSize;   // 本节点权重切片 size（局部）
} NnColMatmulSliceUneven;   // 每个节点一份（数组使用）

typedef struct {
    // —— Q 侧 —— 
    NnUint qDim0;       // 本节点的 q 局部总宽（可保留）
    NnUint qDimStart;   // 本节点在 Q 维的起点
    NnUint qDimLen;     // 新增：本节点在 Q 维的长度（明确代替/补充 qDim0）

    // —— 针对缓存/移位，用于定位 —— 
    NnUint qShift;

    // —— KV 侧 —— 
    NnUint kvDim;       // 全局 kv 维（可保留）
    NnUint kvDim0;      // 本节点 kv 局部（可保留）
    NnUint kvDimStart;  // 起点
    NnUint kvDimLen;    // 新增：长度（替代/补充 kvDim0）

    // —— 其它参数 —— 
    NnUint sliceDim;    // 若你在用它表达“本地处理宽度”，可保留；建议在注释中注明其与 *Len 的关系
    NnUint seqLen;
    NnUint headDim;
    NnUint nKvHeads;
    float  ropeTheta;

    // —— 本节点的 RoPE cache 分配 —— 
    NnSize3D cacheSize; // 局部 cache 尺寸（由上面长度推导）
} NnRopeSliceUneven;

typedef struct {
    // —— 新增：非均匀所需 —— 
    NnUint headStart;  // 本节点负责的 head 起点（全局编号）
    NnUint headLen;    // 本节点负责的 head 数量

    // —— 保留：兼容旧语义 —— 
    NnUint nHeads;     // 全局 head 数
    NnUint nHeads0;    // 本节点 head 局部（可与 headLen 等价；保留以兼容旧 kernel）
    NnSize3D attSize;  // 本节点的注意力中间缓冲尺寸（局部）
} NnMultiHeadAttSliceUneven;

// base enums

enum NnOpCode {
    OP_MERGE_ADD,
    OP_MERGE_SUM,
    OP_EMBEDDING,
    OP_INV_RMS,
    OP_RMS_NORM,
    OP_MATMUL,
    OP_ROPE,
    OP_MULTIHEAD_ATT,
    OP_GELU,
    OP_SILU,
    OP_MUL,
    OP_SCALE,
    OP_CAST,
    OP_REPEAT_Z,
    OP_SHIFT,
    OP_SOFTMAX,
    OP_MOE_GATE,
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

#define N_OP_CODES (OP_SHIFT + 1)
#define N_OP_QUANTS (Q80_F32_F32 + 1)

enum NnPointerSource {
    SRC_PIPE,
    SRC_BUFFER,
};

enum NnPointerType {
    PNTR_RAW,
    PNTR_BATCH,
    PNTR_BATCHED_SLICE
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
    NnSize3D size;
} NnPipeConfig;

typedef struct {
    char *name;
    NnSize3D size;
} NnBufferConfig;

typedef struct {
    NnPointerSource source;
    NnUint pointerIndex;
    NnPointerType type;
} NnPointerConfig;

typedef struct {
    NnOpCode code;
    char *name;
    NnUint index;
    NnPointerConfig input;
    NnPointerConfig output;
    NnSize3D weightSize;
    NnByte *config;
    NnUint configSize;
} NnOpConfig;

typedef struct {
    NnUint pipeIndex;
} NnPreSyncConfig;

typedef struct {
    NnUint pipeIndex;
    NnSyncType syncType;
} NnSyncConfig;

typedef struct  {
    NnUint nOps;
    NnOpConfig *ops;
    NnUint nSyncs;
    NnSyncConfig *syncs;
} NnSegmentConfig;

typedef struct {
    NnUint nBatches;
    NnUint nNodes;
    NnUint nPipes;
    NnPipeConfig *pipes;
    NnUint nPreSyncs;
    NnPreSyncConfig *preSyncs;
} NnNetConfig;

typedef struct {
    NnUint nodeIndex;
    NnUint nBuffers;
    NnBufferConfig *buffers;
    NnUint nSegments;
    NnSegmentConfig *segments;
} NnNodeConfig;

// op configs

typedef struct {
    // empty
} NnEmbeddingOpConfig;

typedef struct {
    float epsilon;
    NnUint nColumns;
} NnInvRmsOpConfig;

typedef struct {
    NnUint invRmsBufferIndex;
    NnUint nColumns;
} NnRmsNormOpConfig;

typedef struct {
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint activeExpertIndexesBufferIndex;
} NnMatmulOpConfig;

typedef struct {
    NnRopeType type;
    NnUint isQ; // Cannot use `bool` here due to GPU memory alignment
    NnUint positionPipeIndex;
    NnUint ropeCacheBufferIndex;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactor;
    NnUint ropeScalingOrigMaxSeqLen;
    NnRopeSlice slice;
} NnRopeOpConfig;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnUint nKvHeads;
    NnUint headDim;
    NnUint seqLen;
    NnUint qSliceD0;
    NnUint kvDim0;
    NnUint positionPipeIndex;
    NnUint queryBufferIndex;
    NnUint keyCacheBufferIndex;
    NnUint valueCacheBufferIndex;
    NnUint attBufferIndex;
} NnMultiHeadAttOpConfig;

typedef struct {
    // empty
} NnMergeAddOpCodeConfig;

typedef struct {
    // empty
} NnMergeSumOpCodeConfig;

typedef struct {
    // empty
} NnSiluOpCodeConfig;

typedef struct {
    NnUint multiplierBufferIndex;
} NnMulOpCodeConfig;

typedef struct {
    NnUint scaleBufferIndex;
} NnScaleOpCodeConfig;

typedef struct {
    // empty
} NnCastOpCodeConfig;

typedef struct {
    // empty
} NnRepeatZOpCodeConfig;

typedef struct {
    NnUint indexPipeIndex;
} NnShiftOpCodeConfig;

typedef struct {
    // empty
} NnSoftmaxOpCodeConfig;

typedef struct {
    NnUint k;
    NnUint normTopk;
    NnUint indexesBufferIndex;
} NnMoeGateOpCodeConfig;

// utility functions

const char *opCodeToString(NnOpCode code);
const char *opQuantTypeToString(NnOpQuantType type);

NnSize getBytes(NnFloatType floatType, NnSize n);
NnSize getBlockSize(NnFloatType floatType);
NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output);
NnSize3D size0();
NnSize3D size1D(NnFloatType floatType, NnUint x);
NnSize3D size2D(NnFloatType floatType, NnUint y, NnUint x);
NnSize3D size3D(NnFloatType floatType, NnUint z, NnUint y, NnUint x);
NnPointerConfig pointerBatchConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerBatchedSliceConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerRawConfig(NnPointerSource source, NnUint index);
bool hasPointerContinuousMemory(NnPointerConfig *config);

void releaseNetConfig(NnNetConfig *netConfig);
void releaseNodeConfig(NnNodeConfig *nodeConfig);

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig);

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    Timer();
    void reset();
    NnUint elapsedMiliseconds();
    NnUint elapsedMicroseconds();
};

// slicers

NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes);
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnRopeSlice sliceRope(NnRopeType type, NnUint qDim, NnUint kvDim, NnUint nKvHeads, NnUint nNodes, NnUint seqLen, NnUint headDim, float ropeTheta, NnUint nodeIndex);
NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches);

// splitters

NnUint splitRowMatmulWeight(NnRowMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);
NnUint splitColMatmulWeight(NnColMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);

// rope

void fullfillRopeCache(const NnRopeOpConfig *config, float *cache);

#endif
