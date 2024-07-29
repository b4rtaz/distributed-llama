#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include <cstddef>
#include <cstdint>
#include "quants.hpp"
#include "commands.hpp"
#include "socket.hpp"

enum TransformerHeaderKey {
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
    WEIGHTS_FLOAT_TYPE = 13,
    ROPE_SCALING_FACTOR = 14,
    ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    ROPE_SCALING_HIGH_FREQ_FACTORY = 16,
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    ROPE_TYPE = 18,
};

struct TransformerFileOldHeader {
    int dim;
    int hiddenDim;
    int nLayers;
    int nHeads;
    int nKvHeads;
    int nExperts;
    int nActiveExperts;
    int vocabSize;
    int seqLen;
};

enum TransformerArchType {
    LLAMA = 0xABCD00,
    GROK1 = 0xABCD01,
    MIXTRAL = 0xABCD02
};

enum TransformerHiddenAct {
    GELU = 0,
    SILU = 1,
};

enum TransformerRopeType {
    ROPE_UNKNOWN = -1,
    ROPE_LLAMA = 0,
    ROPE_FALCON = 1,
    ROPE_LLAMA3_1 = 2,
};

struct TransformerSpec {
    size_t headerSize;
    size_t fileSize;
    int version;
    TransformerArchType archType;
    int dim;
    int nLayers;
    int nHeads;
    int headSize;
    int nKvHeads;
    int nExperts;
    int nActiveExperts;
    unsigned int origSeqLen; // Original model context length
    unsigned int seqLen; // Limited context length by the `--max-seq-len` argument
    int hiddenDim;
    TransformerHiddenAct hiddenAct;
    int kvDim;
    int vocabSize;
    float ropeTheta;
    TransformerRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    int ropeScalingOrigMaxSeqLen;

    FloatType weightsFloatType;
    FloatType bufferFloatType;
    uint8_t nSlices;
};

struct TransformerConfig {
    bool useDiscForKvCache;
};

class TransformerBlock {
public:
    slice_index_t sliceIndex;
    TransformerSpec *spec;
    TransformerConfig* config;

    size_t rmsAttBytes;
    float* rmsAtt;
    size_t rmsFfnBytes;
    float* rmsFfn;
    size_t rmsMoeBytes;
    float* rmsMoe;
    size_t rmsFfn2Bytes;
    float* rmsFfn2;

    MatmulCommand *q0mm;
    MatmulCommand *k0mm;
    MatmulCommand *v0mm;
    MatmulCommand *wo0mm;
    RowMatmulSlice* q0Slice;
    RowMatmulSlice* k0Slice;
    RowMatmulSlice* v0Slice;
    ColMatmulSlice* wo0Slice;

    MatmulCommand *w10mm;
    MatmulCommand *w20mm;
    MatmulCommand *w30mm;
    RowMatmulSlice* w10Slice;
    ColMatmulSlice* w20Slice;
    RowMatmulSlice* w30Slice;

    MatmulCommand* moeRouterMm;
    RowMatmulSlice* moeUpAndGate0Slice;
    RowMatmulSlice* moeDown0Slice;
    MatmulCommand** moeUpMm;
    MatmulCommand** moeGateMm;
    MatmulCommand** moeDownMm;

    float* moeRouterProbs;
    float* expertGate;
    float* expertDown;
    float* hb20;

    KvCacheSlice* kvCacheSlice;
    float* keyCache;
    float* valueCache;
    MultiHeadAttSlice* multiHeadAttSlice;
    float* att;
    float* qo0;

    TransformerBlock(TransformerSpec* spec, TransformerConfig* config, slice_index_t sliceIndex);
    ~TransformerBlock();
};

#define TB_LENGTH 10
#define TB_NO_PAIRS 2

#define TB_UNIT_XB 0
#define TB_UNIT_XB_QUANTIZED 1
#define TB_SLICED_XB2 2
#define TB_SLICED_XB2_QUANTIZED 3
#define TB_SLICED_XBV 4
#define TB_SLICED_XBV_QUANTIZED 5
#define TB_SLICED_HB 6
#define TB_SLICED_HB_QUANTIZED 7
#define TB_UNIT_MOE_INDEXES 8
#define TB_UNIT_MOE_WEIGHTS 9

class TransformerBuffer {
public:
    uint8_t nSlices;
    void** buffers;
    size_t* bufferBytes;

    TransformerBuffer(TransformerSpec* spec);
    ~TransformerBuffer();
    void* getUnit(uint8_t bufferIndex);
    size_t getUnitBytes(uint8_t bufferIndex);
    void* getSliced(uint8_t bufferIndex, slice_index_t sliceIndex);
    size_t getSlicedBytes(uint8_t bufferIndex);
};

class Transformer {
public:
    TransformerSpec* spec;
    TransformerConfig* config;
    TransformerBlock** blocks;
    TransformerBuffer* buffer;
    slice_index_t sliceIndex;

    size_t tokenEmbeddingTableBytes;
    float* tokenEmbeddingTable;
    size_t rmsFinalBytes;
    float* rmsFinal;
    MatmulCommand* wclsMm;

    pos_t pos;
    float rms;
    float* x;
    float* logits;
    RopeSlice* ropeSlice;
    RopeCommand* rope;

    ~Transformer();

    static TransformerSpec loadSpecFromFile(const char* path, const unsigned int nSlices, const unsigned int maxSeqLen, FloatType weightsFloatType, FloatType bufferFloatType);
    static Transformer loadRootFromFile(const char* path, TransformerSpec* spec, TransformerConfig* config, SocketPool* socketPool);
    static Transformer loadRoot(char* data, TransformerSpec* spec, TransformerConfig* config, SocketPool* socketPool);
    static Transformer loadSlice(TransformerSpec* spec, TransformerConfig* config, Socket* socket);

private:
    Transformer(TransformerSpec* spec, TransformerConfig* config, slice_index_t sliceIndex);
};

#endif
