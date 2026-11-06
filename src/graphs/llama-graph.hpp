#ifndef LLAMA_GRAPH_HPP
#define LLAMA_GRAPH_HPP

#include "graph.hpp"

enum LlamaLlmHeaderKey {
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

enum LlamaLlmArchType {
    LLAMA = 0xABCD00,
    QWEN3 = 0xABCD01,
    QWEN3_MOE = 0xABCD02,
};

struct LlamaLlmHeader : public LlmHeader {
    LlamaLlmArchType archType;
    NnUint dim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headDim;
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint hiddenDim;
    NnUint moeHiddenDim;
    NnHiddenAct hiddenAct;
    NnUint qDim;
    NnUint kvDim;
    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnUint ropeScalingOrigMaxSeqLen;
    float normEpsilon;
};

struct LlamaGraph : public LlmGraph {
    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnColMatmulSlice woSlice;
    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;
    NnRowMatmulSlice wclsSlice;
    NnUint xPipeIndex;
    NnSize3D tokenEmbeddingSize;
    NnSize3D rmsNormSize;
    NnSize3D qkRmsNormSize;
    NnSize3D moeGateSize;
};

#endif
