#include <stdio.h>
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
    int dim;
    int nLayers;
    int nHeads;
    int headSize;
    int nKvHeads;
    int seqLen;
    int hiddenDim;
    int kvDim;
    int vocabSize;
    int sliceCount;
};

SharedBuffer* createTransformerSharedBuffer(TransformerSpec* spec);

class TransformerBlockFragment {
protected:
    int layerIndex;
    int sliceIndex;
    TransformerSpec* spec;
    SharedBuffer *sharedBuffer;

public:
    TransformerBlockFragment(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
};

class TransformerBlockQkv: public TransformerBlockFragment {
public:
    MatMulSlice *qSlice;
    MatMulSlice *kSlice;
    MatMulSlice *vSlice;

    TransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    virtual ~TransformerBlockQkv();

    virtual void readWeights(float *qWeights, float *kWeights, float *vWeights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockQkv: public TransformerBlockQkv {
private:
    float *qWeights0;
    float *kWeights0;
    float *vWeights0;

public:
    NativeTransformerBlockQkv(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~NativeTransformerBlockQkv();

    void readWeights(float *qWeights, float *kWeights, float *vWeights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlockAtt: public TransformerBlockFragment {
public:
    MatMulSlice *woSlice;

    TransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    virtual ~TransformerBlockAtt();

    virtual void readWeights(float *woWeights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockAtt: public TransformerBlockAtt {
private:
    float *woWeights0;

public:
    NativeTransformerBlockAtt(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~NativeTransformerBlockAtt();

    void readWeights(float *woWeights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlockFfn: public TransformerBlockFragment {
public:
    MatMulSlice *w1Slice;
    MatMulSlice *w3Slice;

    TransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    virtual ~TransformerBlockFfn();

    virtual void readWeights(float *w1Weights, float *w3Weights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockFfn: public TransformerBlockFfn {
private:
    float *w1Weights0;
    float *w3Weights0;
    float *hb20;

public:
    NativeTransformerBlockFfn(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~NativeTransformerBlockFfn();

    void readWeights(float *w1Weights, float *w3Weights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlockFfn2: public TransformerBlockFragment {
public:
    MatMulSlice *w2Slice;

    TransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    virtual ~TransformerBlockFfn2();

    virtual void readWeights(float *w2Weights) = 0;
    virtual void beginForwarding() = 0;
    virtual void waitForEnd() = 0;
};

class NativeTransformerBlockFfn2: public TransformerBlockFfn2 {
private:
    float *w2Weights0;

public:
    NativeTransformerBlockFfn2(int layerIndex, int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~NativeTransformerBlockFfn2();

    void readWeights(float *w2Weights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlock {
private:
    int layerIndex;
    TransformerSpec* spec;
    SharedBuffer *sharedBuffer;
    TransformerBlockQkv **qkvs;
    TransformerBlockAtt **atts;
    TransformerBlockFfn **ffns;
    TransformerBlockFfn2 **ffn2s;
public:
    float* rmsAttWeight; // (dim)
    float* rmsFfnWeight; // (dim)

    float* xb2; // (dim)
    float* hb; // (hidden_dim)
    float* q; // (dim)
    float* keyCache; // (seq_len, kv_dim)
    float* valueCache; // (seq_len, kv_dim)
    float* att; // (n_heads, seq_len)

    TransformerBlock(int layerIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~TransformerBlock();

    long readWeights(char *wd);
    void forward(int pos, float* x);
};

class Transformer {
private:
    TransformerSpec* spec;
    SharedBuffer* sharedBuffer;
    TransformerBlock** blocks;

    float* x;
    float* token_embedding_table;
    float* rms_final_weight;
    float* wcls;
public:
    float* logits;
    Transformer(TransformerSpec* spec, SharedBuffer* sharedBuffer);
    ~Transformer();

    long readWeights(char* wd, bool sharedWeights);
    void forward(int token, int pos);
};

#endif
