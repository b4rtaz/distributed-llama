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

struct TransformerFileHeader {
    int dim;
    int hiddenDim;
    int nLayers;
    int nHeads;
    int nKvHeads;
    int vocabSize;
    int seqLen;
};

struct TransformerSpec {
    size_t fileSize;
    int dim;
    int nLayers;
    int nHeads;
    int headSize;
    int nKvHeads;
    int seqLen;
    int hiddenDim;
    int kvDim;
    bool sharedWeights;
    int vocabSize;

    FloatType floatType;
    uint8_t nSlices;
};

class TransformerBlock {
public:
    uint8_t sliceIndex;

    size_t rmsAttBytes;
    float* rmsAtt;
    size_t rmsFfnBytes;
    float* rmsFfn;

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

    float* keyCache;
    float* valueCache;
    float* att;
    float* hb20;

    TransformerBlock(TransformerSpec* spec, uint8_t sliceIndex);
    ~TransformerBlock();
};

#define TB_LENGTH 6
#define TB_UNIT_XB 0
#define TB_SLICED_XB2 1
#define TB_SLICED_Q 2
#define TB_SLICED_K 3
#define TB_SLICED_V 4
#define TB_SLICED_HB 5

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

    static TransformerSpec loadSpecFromFile(const char* path, const unsigned int nSlices, FloatType type);
    static Transformer loadRootFromFile(const char* path, TransformerSpec* spec, SocketPool* socketPool);
    static Transformer loadRoot(char* data, TransformerSpec* spec, SocketPool* socketPool);
    static Transformer loadSlice(TransformerSpec* spec, Socket* socket);

private:
    Transformer(TransformerSpec* spec, uint8_t sliceIndex);
};

#endif
