#include <cstdio>
#include <pthread.h>
#include "shared-buffer.hpp"
#include "matmul.hpp"

#ifndef transformer_hpp
#define transformer_hpp

#define SB_LENGTH 7
#define SB_UNIT_XB 0
#define SB_UNIT_HH 1
#define SB_SLICED_XB2 2
#define SB_SLICED_Q 3
#define SB_SLICED_K 4
#define SB_SLICED_V 5
#define SB_SLICED_HB 6

class TransformerSpec {
public:
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
    int sliceCount;
};

class RemoteClient {
public:
    virtual void createFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type, char* data, size_t bytes) = 0;
    virtual void forwardFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type) = 0;
    virtual void sendBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes) = 0;
    virtual void readBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes) = 0;
    virtual void dumpStatistics() = 0;
};

#define TRANSFORMER_BLOCK_QKV 0
#define TRANSFORMER_BLOCK_ATT 1
#define TRANSFORMER_BLOCK_FFN 2
#define TRANSFORMER_BLOCK_FFN2 3

class TransformerState {
public:
    virtual char* getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) = 0;
    virtual char* getUnitBuffer(uint8_t bufferIndex) = 0;
    virtual void readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) = 0;
    virtual void sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex) = 0;
    virtual void sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) = 0;
};

class NativeTransformerState: public TransformerState {
private:
    SharedBuffer* buffer;
public:
    NativeTransformerState(SharedBuffer* buffer);
    char* getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    char* getUnitBuffer(uint8_t bufferIndex);
    void readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
};

class RemoteTransformerState: public TransformerState {
private:
    SharedBuffer* buffer;
    RemoteClient* client;
public:
    RemoteTransformerState(SharedBuffer* buffer, RemoteClient* client);
    ~RemoteTransformerState();
    char* getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    char* getUnitBuffer(uint8_t bufferIndex);
    void readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
};

SharedBuffer* initSharedBuffer(TransformerSpec* spec);

//
// TransformerFragment
//

class TransformerFragment {
protected:
    int layerIndex;
    int sliceIndex;
    TransformerSpec* spec;
    TransformerState* state;

public:
    TransformerFragment(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
};

//
// TransformerBlockQkv
//

class TransformerBlockQkv: public TransformerFragment {
public:
    MatMulSlice* qSlice;
    MatMulSlice* kSlice;
    MatMulSlice* vSlice;

    TransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    virtual ~TransformerBlockQkv();

    virtual void readWeights(char* qWeights, char* kWeights, char* vWeights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockQkv: public TransformerBlockQkv {
public:
    char* qWeights0;
    char* kWeights0;
    char* vWeights0;

    NativeTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    ~NativeTransformerBlockQkv();

    void readWeights(char* qWeights, char* kWeights, char* vWeights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockQkv: public TransformerBlockQkv {
private:
    RemoteClient* client;
public:
    RemoteTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client);

    void readWeights(char* qWeights, char* kWeights, char* vWeights);
    void beginForwarding();
    void waitForEnd();
};

//
// TransformerBlockAtt
//

class TransformerBlockAtt: public TransformerFragment {
public:
    MatMulSlice* woSlice;

    TransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    virtual ~TransformerBlockAtt();

    virtual void readWeights(char* woWeights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockAtt: public TransformerBlockAtt {
public:
    char* woWeights0;

    NativeTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    ~NativeTransformerBlockAtt();

    void readWeights(char* woWeights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockAtt: public TransformerBlockAtt {
private:
    RemoteClient* client;
public:
    RemoteTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client);

    void readWeights(char* woWeights);
    void beginForwarding();
    void waitForEnd();
};

//
// TransformerBlockFfn
//

class TransformerBlockFfn: public TransformerFragment {
public:
    MatMulSlice* w1Slice;
    MatMulSlice* w3Slice;

    TransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    virtual ~TransformerBlockFfn();

    virtual void readWeights(char* w1Weights, char* w3Weights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockFfn: public TransformerBlockFfn {
private:
    float *hb20;
public:
    char* w1Weights0;
    char* w3Weights0;

    NativeTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    ~NativeTransformerBlockFfn();

    void readWeights(char* w1Weights, char* w3Weights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockFfn: public TransformerBlockFfn {
private:
    RemoteClient* client;
public:
    RemoteTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client);

    void readWeights(char* w1Weights, char* w3Weights);
    void beginForwarding();
    void waitForEnd();
};

//
// TransformerBlockFfn2
//

class TransformerBlockFfn2: public TransformerFragment {
public:
    MatMulSlice* w2Slice;

    TransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    virtual ~TransformerBlockFfn2();

    virtual void readWeights(char* w2Weights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockFfn2: public TransformerBlockFfn2 {
public:
    char* w2Weights0;

    NativeTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state);
    ~NativeTransformerBlockFfn2();

    void readWeights(char* w2Weights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockFfn2: public TransformerBlockFfn2 {
private:
    RemoteClient* client;
public:
    RemoteTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* client);

    void readWeights(char* w2Weights);
    void beginForwarding();
    void waitForEnd();
};

//
// TransformerBlock
//

struct TransformerBlockThreadInfo {
    pthread_t handler;
    int sliceIndex;
    int step;
    TransformerBlockQkv* qkv;
    TransformerBlockAtt* att;
    TransformerBlockFfn* ffn;
    TransformerBlockFfn2* ffn2;
};

class TransformerBlock {
private:
    int layerIndex;
    TransformerSpec* spec;
    TransformerState* state;
    TransformerBlockQkv **qkvs;
    TransformerBlockAtt **atts;
    TransformerBlockFfn **ffns;
    TransformerBlockFfn2 **ffn2s;
    TransformerBlockThreadInfo *threadInfos;
public:
    float* rmsAttWeight; // (dim)
    float* rmsFfnWeight; // (dim)

    float* xb2; // (dim)
    float* hb; // (hidden_dim)
    float* q; // (dim)
    float* keyCache; // (seq_len, kv_dim)
    float* valueCache; // (seq_len, kv_dim)
    float* att; // (n_heads, seq_len)

    TransformerBlock(int layerIndex, TransformerSpec* spec, TransformerState* state, RemoteClient* clientOrNull);
    ~TransformerBlock();

    long readWeights(char* wd);
    void forward(int pos, float* x);
};

class Transformer {
public:
    TransformerSpec* spec;
    RemoteClient* clientOrNull;
private:
    TransformerState* state;
    TransformerBlock** blocks;

    float* x;
    float* token_embedding_table;
    float* rms_final_weight;
    float* wcls;
public:
    float* logits;
    Transformer(TransformerSpec* spec, TransformerState* state, RemoteClient* clientOrNull);
    ~Transformer();

    long readWeights(char* wd);
    void forward(int token, int pos);
};

void loadTransformerSpec(TransformerSpec* spec, const char* path, FloatType type, int sliceCount);
void loadTransformer(Transformer** transformerOut, TransformerSpec* spec, const char* path, RemoteClient* clientOrNull);

#endif
