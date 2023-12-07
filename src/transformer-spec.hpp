#ifndef transformer_spec_hpp
#define transformer_spec_hpp

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

#endif
