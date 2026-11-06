#ifndef QWEN35_GRAPH_HPP
#define QWEN35_GRAPH_HPP

#include "graph.hpp"

enum Qwen35LlmHeaderKey {
    QWEN35_VERSION = 0,
    QWEN35_ARCH_TYPE = 1,
    QWEN35_DIM = 2,
    QWEN35_HIDDEN_DIM = 3,
    QWEN35_N_LAYERS = 4,
    QWEN35_N_HEADS = 5,
    QWEN35_N_KV_HEADS = 6,
    QWEN35_VOCAB_SIZE = 7,
    QWEN35_SEQ_LEN = 8,
    QWEN35_ROPE_THETA = 9,
    QWEN35_WEIGHT_FLOAT_TYPE = 10,
    QWEN35_HEAD_DIM = 11,
    QWEN35_ROTARY_DIM = 12,
    QWEN35_NORM_EPSILON = 13,
    QWEN35_FULL_ATTENTION_INTERVAL = 14,
    QWEN35_LINEAR_CONV_KERNEL_DIM = 15,
    QWEN35_LINEAR_KEY_HEAD_DIM = 16,
    QWEN35_LINEAR_VALUE_HEAD_DIM = 17,
    QWEN35_LINEAR_N_KEY_HEADS = 18,
    QWEN35_LINEAR_N_VALUE_HEADS = 19,
};

enum Qwen35LlmArchType {
    QWEN35 = 0xABCE01,
};

struct Qwen35LlmHeader : public LlmHeader {
    Qwen35LlmArchType archType;
    NnUint dim;
    NnUint hiddenDim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headDim;
    NnUint rotaryDim;
    NnUint qDim;
    NnUint kvDim;
    float ropeTheta;
    float normEpsilon;
    NnUint fullAttentionInterval;
    NnUint linearConvKernelDim;
    NnUint linearKeyHeadDim;
    NnUint linearValueHeadDim;
    NnUint linearNKeyHeads;
    NnUint linearNValueHeads;
};

struct Qwen35Graph : public LlmGraph {
    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnRowMatmulSlice gateSlice;
    NnColMatmulSlice woSlice;
    NnRowMatmulSlice linearQSlice;
    NnRowMatmulSlice linearKSlice;
    NnRowMatmulSlice linearVSlice;
    NnRowMatmulSlice linearZSlice;
    NnRowMatmulSlice linearASlice;
    NnRowMatmulSlice linearBSlice;
    NnColMatmulSlice linearOutSlice;
    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;
    NnRowMatmulSlice wclsSlice;
    NnUint xPipeIndex;
    NnSize3D tokenEmbeddingSize;
    NnSize3D rmsNormSize;
    NnSize3D qkRmsNormSize;
};

#endif
