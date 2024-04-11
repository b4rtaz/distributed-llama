#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include <cstddef>
#include <cstdint>
#include "quants.hpp"
#include "socket.hpp"

class MatmulSlice {
public:
    FloatType type;
    int nSlices;
    int d0;
    int n;
    size_t bytes;
    size_t sliceBytes;

    MatmulSlice(FloatType type, int nSlices, int n, int d);
    size_t splitWeights(uint8_t sliceIndex, char* weights, char* weights0);
    long mergeOutputs(uint8_t sliceIndex, float* output, float* output0);
};

enum TransformerArchType {
    LLAMA2 = 0xABCD00,
    GROK1 = 0xABCD01
};

struct TransformerFileHeader {
    TransformerArchType archType;
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

struct TransformerSpec {
    size_t fileSize;
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
    int kvDim;
    int vocabSize;

    FloatType weightsFloatType;
    FloatType bufferFloatType;
    uint8_t nSlices;
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
    MatmulSlice* q0Slice;
    char* k0;
    MatmulSlice* k0Slice;
    char* v0;
    MatmulSlice* v0Slice;
    char* wo0;

    MatmulSlice* wo0Slice;
    char* w10;
    MatmulSlice* w10Slice;
    char* w20;
    MatmulSlice* w20Slice;
    char* w30;
    MatmulSlice* w30Slice;


    char* moeRouter;
    size_t moeRouterBytes;
    MatmulSlice* moeUpAndGate0Slice;
    char** moeUp;
    char** moeGate;
    MatmulSlice* moeDown0Slice;
    char** moeDown;
    float* moeRouterProbs;

    float* expertGate;
    float* expertDown;

    float* hb20;

    float* keyCache;
    float* valueCache;
    float* att;

    TransformerBlock(TransformerSpec* spec, uint8_t sliceIndex);
    ~TransformerBlock();
};

#define TB_LENGTH 14
#define TB_NO_PAIRS 2

#define TB_UNIT_XB 0
#define TB_UNIT_XB_QUANTIZED 1
#define TB_SLICED_XB2 2
#define TB_SLICED_XB2_QUANTIZED 3
#define TB_SLICED_Q 4
#define TB_SLICED_Q_QUANTIZED 5
#define TB_SLICED_K 6
#define TB_SLICED_K_QUANTIZED 7
#define TB_SLICED_V 8
#define TB_SLICED_V_QUANTIZED 9
#define TB_SLICED_HB 10
#define TB_SLICED_HB_QUANTIZED 11
#define TB_UNIT_MOE_INDEXES 12
#define TB_UNIT_MOE_WEIGHTS 13

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

    float rms;
    int pos;
    float* x;
    float* logits;

    ~Transformer();

    static TransformerSpec loadSpecFromFile(const char* path, const unsigned int nSlices, FloatType weightsFloatType, FloatType bufferFloatType);
    static Transformer loadRootFromFile(const char* path, TransformerSpec* spec, SocketPool* socketPool);
    static Transformer loadRoot(char* data, TransformerSpec* spec, SocketPool* socketPool);
    static Transformer loadSlice(TransformerSpec* spec, Socket* socket);

private:
    Transformer(TransformerSpec* spec, uint8_t sliceIndex);
};

#endif
