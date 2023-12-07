#include "transformer-spec.hpp"
#include "transformer-block.hpp"
#include "shared-buffer.hpp"

#ifndef transformer_hpp
#define transformer_hpp

class Transformer {
private:
    TransformerSpec* spec;
    SharedBuffer* sharedBuffer;
    TransformerBlock** blocks;

    float* x;
    float* token_embedding_table;
    float* rms_final_weight;
    float* wcls;
    float* logits;
public:
    Transformer(TransformerSpec* spec, SharedBuffer* sharedBuffer);
    ~Transformer();

    void readWeights(char* wd);
    void forward(int token, int pos);
};

#endif
