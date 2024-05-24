#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include <cstddef>
#include <cstdint>
#include "quants.hpp"
#include "socket.hpp"

typedef unsigned short pos_t;

class MatmulSlice {
public:
    size_t bytes;
    size_t sliceBytes;
    virtual size_t splitWeights(uint8_t sliceIndex, char* weights, char* weights0) = 0;
};

class RowMatmulSlice : public MatmulSlice {
public:
    FloatType type;
    int nSlices;
    int n;
    int d0;

    RowMatmulSlice(FloatType type, int nSlices, int n, int d);
    size_t splitWeights(uint8_t sliceIndex, char* weights, char* weights0);
    unsigned int dOffset(uint8_t sliceIndex);
};

class ColMatmulSlice : public MatmulSlice {
public:
    FloatType type;
    int nSlices;
    int n;
    int n0;
    int d;

    ColMatmulSlice(FloatType type, int nSlices, int n, int d);
    size_t splitWeights(uint8_t sliceIndex, char* weights, char* weights0);
};

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
    int seqLen;
    int hiddenDim;
    TransformerHiddenAct hiddenAct;
    int kvDim;
    int vocabSize;
    float ropeTheta;

    FloatType weightsFloatType;
    FloatType bufferFloatType;
    uint8_t nSlices;
};

class RopeSlice {
public:
    unsigned int qDim0;
    unsigned int qDimStart;
    unsigned int qDimEnd;
    unsigned int qShift;
    unsigned int kvDim0;
    unsigned int kvDimStart;
    unsigned int sliceDim;
    TransformerSpec* spec;
    RopeSlice(TransformerSpec* spec, uint8_t sliceIndex);
    virtual ~RopeSlice();
    virtual void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex) = 0;
};

class LlamaRopeSlice : public RopeSlice {
private:
    float* cache;
public:
    LlamaRopeSlice(TransformerSpec* spec, uint8_t sliceIndex);
    ~LlamaRopeSlice();
    void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex);
};

class FalconRopeSlice : public RopeSlice {
public:
    using RopeSlice::RopeSlice;
    void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex);
};

class KvCacheSlice {
public:
    unsigned int kvDim0;
    float* keyCache;
    float* valueCache;
    KvCacheSlice(unsigned int kvDim, unsigned int seqLen, unsigned int nSlices);
    ~KvCacheSlice();
};

class MultiHeadAttSlice {
public:
    unsigned int nHeads0;
    float* att;
    MultiHeadAttSlice(unsigned int nHeads, unsigned int seqLen, unsigned int nSlices, uint8_t sliceIndex);
    ~MultiHeadAttSlice();
};

class TransformerBlock {
public:
    uint8_t sliceIndex;
    TransformerSpec *spec;

    size_t rmsAttBytes;
    float* rmsAtt;
    size_t rmsFfnBytes;
    float* rmsFfn;
    size_t rmsMoeBytes;
    float* rmsMoe;
    size_t rmsFfn2Bytes;
    float* rmsFfn2;

    char* q0;
    RowMatmulSlice* q0Slice;
    char* k0;
    RowMatmulSlice* k0Slice;
    char* v0;
    RowMatmulSlice* v0Slice;
    char* wo0;
    ColMatmulSlice* wo0Slice;

    char* w10;
    RowMatmulSlice* w10Slice;
    char* w20;
    RowMatmulSlice* w20Slice;
    char* w30;
    RowMatmulSlice* w30Slice;

    char* moeRouter;
    size_t moeRouterBytes;
    RowMatmulSlice* moeUpAndGate0Slice;
    char** moeUp;
    char** moeGate;
    RowMatmulSlice* moeDown0Slice;
    char** moeDown;
    float* moeRouterProbs;

    float* expertGate;
    float* expertDown;

    float* hb20;

    KvCacheSlice* kvCacheSlice;
    MultiHeadAttSlice* multiHeadAttSlice;
    float* qo0;

    TransformerBlock(TransformerSpec* spec, uint8_t sliceIndex);
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
    char** buffers;
    size_t* bufferBytes;

    TransformerBuffer(TransformerSpec* spec);
    ~TransformerBuffer();
    char* getUnit(uint8_t bufferIndex);
    size_t getUnitBytes(uint8_t bufferIndex);
    char* getSliced(uint8_t bufferIndex, uint8_t sliceIndex);
    size_t getSlicedBytes(uint8_t bufferIndex);
};

class Transformer {
public:
    TransformerSpec* spec;
    TransformerBlock** blocks;
    TransformerBuffer* buffer;
    uint8_t sliceIndex;

    size_t tokenEmbeddingTableBytes;
    char* tokenEmbeddingTable;
    size_t rmsFinalBytes;
    char* rmsFinal;
    size_t wclsBytes;
    char* wcls;

    pos_t pos;
    float rms;
    float* x;
    float* logits;
    RopeSlice* ropeSlice;

    ~Transformer();

    static TransformerSpec loadSpecFromFile(const char* path, const unsigned int nSlices, FloatType weightsFloatType, FloatType bufferFloatType);
    static Transformer loadRootFromFile(const char* path, TransformerSpec* spec, SocketPool* socketPool);
    static Transformer loadRoot(char* data, TransformerSpec* spec, SocketPool* socketPool);
    static Transformer loadSlice(TransformerSpec* spec, Socket* socket);

private:
    Transformer(TransformerSpec* spec, uint8_t sliceIndex);
};

#endif
