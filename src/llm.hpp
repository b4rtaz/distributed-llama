#ifndef LLM_HPP
#define LLM_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network.hpp"

enum LlmHeaderKey {
    VERSION = 0,
    ARCH_TYPE = 1,
    DIM = 2,
    HIDDEN_DIM = 3,
    N_LAYERS = 4,
    N_HEADS = 5,
    N_KV_HEADS = 6,
    N_EXPERTS = 7,
    N_ACTIVE_EXPERTS = 8,
    VOCAB_SIZE = 9,
    SEQ_LEN = 10,
    HIDDEN_ACT = 11,
    ROPE_THETA = 12,
    WEIGHT_FLOAT_TYPE = 13,
    ROPE_SCALING_FACTOR = 14,
    ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    ROPE_SCALING_HIGH_FREQ_FACTORY = 16,
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    ROPE_TYPE = 18,
    HEAD_DIM = 19,
    NORM_EPSILON = 20,
    MOE_HIDDEN_DIM = 21,
};

enum LlmHiddenAct {
    HIDDEN_ACT_GELU,
    HIDDEN_ACT_SILU,
};

enum LlmArchType {
    LLAMA = 0xABCD00,
    QWEN3 = 0xABCD01,
    QWEN3_MOE = 0xABCD02,
};

typedef struct {
    NnSize headerSize;
    NnSize fileSize;
    int version;
    LlmArchType archType;
    NnUint dim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headDim;
    NnUint nKvHeads;
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint origSeqLen; // Original model context length
    NnUint seqLen; // Limited context length by the `--max-seq-len` argument
    NnUint hiddenDim;
    NnUint moeHiddenDim;
    LlmHiddenAct hiddenAct;
    NnUint qDim;
    NnUint kvDim;
    NnUint vocabSize;
    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnUint ropeScalingOrigMaxSeqLen;
    float normEpsilon;

    NnFloatType weightType;
    NnFloatType syncType;
} LlmHeader;

typedef struct {
    LlmHeader *header;
    NnNetConfig netConfig;
    NnNodeConfig *nodeConfigs;
    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnColMatmulSlice woSlice;
    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;
    NnRowMatmulSlice wclsSlice;
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;
    NnSize3D tokenEmbeddingSize;
    NnSize3D rmsNormSize;
    NnSize3D qkRmsNormSize;
    NnSize3D moeGateSize;
} LlmNet;

typedef struct {
    LlmHeader *header;
    NnNetConfig netConfig;
    NnNodeConfig *nodeConfigs;      // [nNodes]

    // Attention 前投影（行切：按输入维 D_in 切）
    NnRowMatmulSliceUneven *qSlices;      // [nNodes] for W_q
    NnRowMatmulSliceUneven *kSlices;      // [nNodes] for W_k
    NnRowMatmulSliceUneven *vSlices;      // [nNodes] for W_v

    // Attention 输出投影（列切：按输出维 D_out 切）
    NnColMatmulSliceUneven *woSlices;     // [nNodes] for W_o

    // FFN（w1/w3 行切；w2 列切）
    NnRowMatmulSliceUneven *w1Slices;     // [nNodes] for W1 (D_in -> ffDim)
    NnColMatmulSliceUneven *w2Slices;     // [nNodes] for W2 (ffDim -> D_out)
    NnRowMatmulSliceUneven *w3Slices;     // [nNodes] for W3 (D_in -> ffDim)

    // 词表投影（通常按行切 vocab 维）
    NnRowMatmulSliceUneven *wclsSlices;   // [nNodes] for W_cls

    // 多头注意力与缓存的分片（按 head/kv 维非均匀）
    NnMultiHeadAttSliceUneven *mhaSlices; // [nNodes]  每个节点的 headStart / nHeads_node / attSize 等
    NnKvCacheSliceUneven     *kvSlices;   // [nNodes]  每个节点的 KV 维/容量切片
    NnRopeSliceUneven        *ropeSlices; // [nNodes]  每个节点的 RoPE cache 切片（由 head 切映射得到）

    // ---------- ② Pipe 索引（保持原有） ----------
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;

    NnUint zqPipeIndex;
    NnSize3D tokenEmbeddingSize;
    NnSize3D rmsNormSize;
    NnSize3D qkRmsNormSize;
    NnSize3D moeGateSize;

} LlmNetUneven;  


LlmHeader loadLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);
void printLlmHeader(LlmHeader *header);
LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches);
void releaseLlmNet(LlmNet *net);
void loadLlmNetWeight(const char* path, LlmNet *net, NnRootWeightLoader *loader);

#endif