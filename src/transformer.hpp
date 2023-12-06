#include <stdio.h>
#include "shared-buffer.hpp"
#include "matmul.hpp"

#define SB_LENGTH 6
#define SB_X 0
#define SB_XB 1
#define SB_Q 2
#define SB_K 3
#define SB_V 4
#define SB_XB2 5

class TransformerSpec {
public:
    int dim;
    int n_heads;
    int head_size;
    int n_kv_heads;
    int seq_len;
    int hidden_dim;
    int kv_dim;
    int vocab_size;

    int sliceCount;
};

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
    float *w2Weights0;
    float *w3Weights0;

    float *hb0;
    float *hb20;

public:
    MatMulSlice *w1Slice;
    MatMulSlice *w2Slice;
    MatMulSlice *w3Slice;

    TransformerBlockFfn(int sliceIndex, TransformerSpec* spec, SharedBuffer *sharedBuffer);
    ~TransformerBlockFfn();

    void readWeights(float *w1Weights, float *w2Weights, float *w3Weights);
    void beginForwarding();
    void waitForEnd();
};

class TransformerBlock {
private:
    TransformerSpec* spec;
    SharedBuffer *sharedBuffer;
    TransformerBlockQkv **qkvs;
    TransformerBlockAtt **atts;
public:
    float* rms_att_weight; // (dim) rmsnorm weights
    float* rms_ffn_weight; // (dim)
    // weights for matmuls. note dim == n_heads * head_size
    //float* wq; // (dim, n_heads * head_size)
    //float* wk; // (dim, n_kv_heads * head_size)
    //float* wv; // (dim, n_kv_heads * head_size)
    //float* wo; // (n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (hidden_dim, dim)
    float* w2; // (dim, hidden_dim)
    float* w3; // (hidden_dim, dim)

    //float* xb; // (dim)
    float* xb2; // (dim)
    float* hb; // (hidden_dim)
    float* hb2; // (hidden_dim)
    float* q; // (dim)
    float* key_cache; // (seq_len, kv_dim)
    float* value_cache; // (seq_len, kv_dim)
    float* att; // (n_heads, seq_len)
    float* logits; // (vocab_size)

    TransformerBlock(TransformerSpec* spec, SharedBuffer *sharedBuffer, TransformerBlockQkv **qkvs, TransformerBlockAtt **atts);
    ~TransformerBlock();

    void readWeights(FILE *f);
    void forward(int pos);
};
