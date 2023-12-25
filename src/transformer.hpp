#include <cstdio>
#include <cstdint>
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

class TransformerConfig {
public:
    int nThread;
    int sliceCount;
    char** sliceHosts;
    int* slicePorts;
};

#define ACTION_HELLO 0
#define ACTION_CREATE_FRAGMENT 1
#define ACTION_FORWARD_FRAGMENT 2
#define ACTION_SEND_BUFFER 3

class RemoteClient {
private:
    int sliceCount;
    int* clientSockets;
    long* waitBufferTime;
    long* sendBufferTime;
    long* readBufferTime;
public:
    RemoteClient(TransformerSpec* spec, TransformerConfig* config);
    ~RemoteClient();
    void createFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type, char* weights, size_t bytes);
    void forwardFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type);
    void sendBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes);
    void readBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes);
    void dumpStatistics();
private:
    void sendBytes(uint8_t sliceIndex, void* data, size_t bytes);
    void readBytes(uint8_t sliceIndex, void* data, size_t bytes);
};

#define TRANSFORMER_BLOCK_QKV 0
#define TRANSFORMER_BLOCK_ATT 1
#define TRANSFORMER_BLOCK_FFN 2
#define TRANSFORMER_BLOCK_FFN2 3

class TransformerState {
public:
    virtual bool isRemote() = 0;
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
    bool isRemote();
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
    bool isRemote();
    void createFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type, char* weights, size_t bytes);
    void forwardFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type);
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

public:
    TransformerFragment(int layerIndex, int sliceIndex, TransformerSpec* spec);
};

//
// TransformerBlockQkv
//

class TransformerBlockQkv: public TransformerFragment {
public:
    MatMulSlice* qSlice;
    MatMulSlice* kSlice;
    MatMulSlice* vSlice;

    TransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec);
    virtual ~TransformerBlockQkv();

    virtual void readWeights(char* qWeights, char* kWeights, char* vWeights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockQkv: public TransformerBlockQkv {
private:
    TransformerState* state;
    TransformerConfig* config;
public:
    char* qWeights0;
    char* kWeights0;
    char* vWeights0;

    NativeTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState* state);
    ~NativeTransformerBlockQkv();

    void readWeights(char* qWeights, char* kWeights, char* vWeights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockQkv: public TransformerBlockQkv {
private:
    RemoteTransformerState* state;
public:
    RemoteTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, RemoteTransformerState* state);

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

    TransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec);
    virtual ~TransformerBlockAtt();

    virtual void readWeights(char* woWeights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockAtt: public TransformerBlockAtt {
private:
    TransformerState* state;
    TransformerConfig* config;
public:
    char* woWeights0;

    NativeTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState* state);
    ~NativeTransformerBlockAtt();

    void readWeights(char* woWeights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockAtt: public TransformerBlockAtt {
private:
    RemoteTransformerState* state;
public:
    RemoteTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, RemoteTransformerState* state);

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

    TransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec);
    virtual ~TransformerBlockFfn();

    virtual void readWeights(char* w1Weights, char* w3Weights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockFfn: public TransformerBlockFfn {
private:
    TransformerState* state;
    TransformerConfig* config;
    float *hb20;
public:
    char* w1Weights0;
    char* w3Weights0;

    NativeTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState* state);
    ~NativeTransformerBlockFfn();

    void readWeights(char* w1Weights, char* w3Weights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockFfn: public TransformerBlockFfn {
private:
    RemoteTransformerState* state;
public:
    RemoteTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, RemoteTransformerState* state);

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

    TransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec);
    virtual ~TransformerBlockFfn2();

    virtual void readWeights(char* w2Weights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockFfn2: public TransformerBlockFfn2 {
private:
    TransformerState* state;
    TransformerConfig* config;
public:
    char* w2Weights0;

    NativeTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState* state);
    ~NativeTransformerBlockFfn2();

    void readWeights(char* w2Weights);
    void beginForwarding();
    void waitForEnd();
};

class RemoteTransformerBlockFfn2: public TransformerBlockFfn2 {
private:
    RemoteTransformerState* state;
public:
    RemoteTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, RemoteTransformerState* state);

    void readWeights(char* w2Weights);
    void beginForwarding();
    void waitForEnd();
};

//
// TransformerBlock
//

class TransformerBlock;

struct TransformerBlockThreadInfo {
    pthread_t handler;
    int sliceIndex;
    int step;
    TransformerBlock* block;
};

class TransformerBlock {
public:
    int layerIndex;
    TransformerSpec* spec;
    TransformerState* firstState;
    TransformerBlockQkv **qkvs;
    TransformerBlockAtt **atts;
    TransformerBlockFfn **ffns;
    TransformerBlockFfn2 **ffn2s;
    TransformerBlockThreadInfo *threadInfos;

    float* rmsAttWeight; // (dim)
    float* rmsFfnWeight; // (dim)

    float* xb2; // (dim)
    float* hb; // (hidden_dim)
    float* q; // (dim)
    float* keyCache; // (seq_len, kv_dim)
    float* valueCache; // (seq_len, kv_dim)
    float* att; // (n_heads, seq_len)

    TransformerBlock(int layerIndex, TransformerSpec* spec, TransformerConfig* config, TransformerState** states);
    ~TransformerBlock();

    long readWeights(char* wd);
    void forward(int pos, float* x);
    void runStep(int step);
};

class Transformer {
public:
    TransformerSpec* spec;
    TransformerConfig* config;
    RemoteClient* client;
private:
    TransformerBlock** blocks;

    float* x;
    float* token_embedding_table;
    float* rms_final_weight;
    size_t wclsBytes;
    FloatType wclsFloatType;
    char* wcls;
public:
    float* logits;
    Transformer(TransformerSpec* spec, TransformerConfig* config, TransformerState** states, RemoteClient* client);
    ~Transformer();

    long readWeights(char* wd);
    void forward(int token, int pos);
};

void loadTransformerSpec(TransformerSpec* spec, const char* path, FloatType type, int sliceCount);
void loadTransformer(Transformer** transformerOut, TransformerSpec* spec, TransformerConfig* config, const char* path);

#endif
