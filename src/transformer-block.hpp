#include <stdio.h>
#include "transformer-spec.hpp"
#include "shared-buffer.hpp"
#include "matmul.hpp"

#ifndef transformer_block_hpp
#define transformer_block_hpp

#define SB_LENGTH 7
#define SB_UNIT_XB 0
#define SB_UNIT_HH 1
#define SB_SLICED_XB2 2
#define SB_SLICED_Q 3
#define SB_SLICED_K 4
#define SB_SLICED_V 5
#define SB_SLICED_HB 6

SharedBuffer* createTransformerSharedBuffer(TransformerSpec* spec);

class TransformerBlockFragment {
protected:
    int sliceIndex;
    TransformerSpec* spec;
    SharedBuffer *sharedBuffer;
};

class TransformerBlockQkv: public TransformerBlockFragment {
private:
    float *qWeights0;
    float *kWeights0;
    float *vWeights0;

public:
    MatMulSlice *qSlice;
    MatMulSlice *kSlice;
    MatMulSlice *vSlice;

    TransformerBlockQkv(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~TransformerBlockQkv();

    void readWeights(float *qWeights, float *kWeights, float *vWeights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlockAtt: public TransformerBlockFragment {
private:
    float *woWeights0;
public:
    MatMulSlice *woSlice;

    TransformerBlockAtt(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~TransformerBlockAtt();

    void readWeights(float *woWeights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlockFfn: public TransformerBlockFragment {
private:
    float *w1Weights0;
    float *w3Weights0;
    float *hb20;

public:
    MatMulSlice *w1Slice;
    MatMulSlice *w3Slice;

    TransformerBlockFfn(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~TransformerBlockFfn();

    void readWeights(float *w1Weights, float *w3Weights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlockFfn2: public TransformerBlockFragment {
private:
    float *w2Weights0;

public:
    MatMulSlice *w2Slice;

    TransformerBlockFfn2(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~TransformerBlockFfn2();

    void readWeights(float *w2Weights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlock {
private:
    TransformerSpec* spec;
    SharedBuffer *sharedBuffer;
    TransformerBlockQkv **qkvs;
    TransformerBlockAtt **atts;
    TransformerBlockFfn **ffns;
    TransformerBlockFfn2 **ffn2s;
public:
    float* rmsAttWeight; // (dim) rmsnorm weights
    float* rmsFfnWeight; // (dim)

    float* xb2; // (dim)
    float* hb; // (hidden_dim)
    float* q; // (dim)
    float* keyCache; // (seq_len, kv_dim)
    float* valueCache; // (seq_len, kv_dim)
    float* att; // (n_heads, seq_len)

    TransformerBlock(TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~TransformerBlock();

    long readWeights(char *wd);
    void forward(int pos, float* x);
};

#endif
