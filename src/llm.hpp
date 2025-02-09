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
};

enum LlmHiddenAct {
    HIDDEN_ACT_GELU,
    HIDDEN_ACT_SILU,
};

enum LlmArchType {
    LLAMA = 0xABCD00,
};

typedef struct {
    size_t headerSize;
    size_t fileSize;
    int version;
    LlmArchType archType;
    NnSize dim;
    NnSize nLayers;
    NnSize nHeads;
    NnSize headSize;
    NnSize nKvHeads;
    NnSize nExperts;
    NnSize nActiveExperts;
    NnSize origSeqLen; // Original model context length
    NnSize seqLen; // Limited context length by the `--max-seq-len` argument
    NnSize hiddenDim;
    LlmHiddenAct hiddenAct;
    NnSize kvDim;
    NnSize vocabSize;
    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnSize ropeScalingOrigMaxSeqLen;
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
    NnSize positionPipeIndex;
    NnSize tokenPipeIndex;
    NnSize xPipeIndex;
    NnSize logitsPipeIndex;
    NnSize2D tokenEmbeddingSize;
    NnSize2D rmsNormSize;
} LlmNet;

LlmHeader loadLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);
void printLlmHeader(LlmHeader *header);
LlmNet buildLlmNet(LlmHeader *h, NnSize nNodes, NnSize nBatches);
void releaseLlmNet(LlmNet *net);
void loadLlmNetWeight(const char* path, LlmNet *net, NnRootWeightLoader *loader);

#endif