#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP

#include <cstddef>
#include <cstdint>
#include "quants.hpp"
#include "commands.hpp"
#include "socket.hpp"

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
}; // --> 

enum TransformerArchType { // --> 结构类型
    LLAMA = 0xABCD00,
    GROK1 = 0xABCD01,
    MIXTRAL = 0xABCD02
}; 

enum TransformerHiddenAct { // --> 枚举类,激活层选择
    GELU = 0,
    SILU = 1,
};

struct TransformerSpec { // --> 定义一个Transformer需要的相关信息的结构体
    size_t headerSize;
    size_t fileSize;
    int version;
    TransformerArchType archType; // -->模型具体架构
    int dim; // -->模型输入和输出的向量维度
    int nLayers; // -->Transformer中的编码器或解码器的个数
    int nHeads; // -->MHA中的Head个数
    int headSize; // -->MHA中每个Head的维度大小 = 总维度dim / Head个数nHeads
    int nKvHeads; // -->Kv个数
    int nExperts; // -->专家数量,用于稀疏化专家模型例如MoE
    int nActiveExperts; // --> 每次推理中活跃的专家数量
    int seqLen; // --> 输入序列的最大长度
    int hiddenDim; // --> 隐藏层维度
    TransformerHiddenAct hiddenAct; // --> 激活层类型
    int kvDim; // --> Kv维度大小
    int vocabSize; // --> 词汇表的大小，表示模型可以处理的不同词汇的数量。用于嵌入层（embedding layer）的输入和输出维度。
    float ropeTheta; // -->控制旋转编码的频率

    FloatType weightsFloatType;
    FloatType bufferFloatType;
    // 量化选择
    uint8_t nSlices;

    // 定义一个memoryBudgetArray
    std::vector<int> memoryBudgetArray;
};

class TransformerBlock { // -->给Q,K,V,W分配空间
public:
    slice_index_t sliceIndex; // unsigned char
    TransformerSpec *spec; // 与Transformer模型相关的配置参数

    size_t rmsAttBytes;
    float* rmsAtt;
    size_t rmsFfnBytes;
    float* rmsFfn;
    size_t rmsMoeBytes;
    float* rmsMoe;
    size_t rmsFfn2Bytes;
    float* rmsFfn2;

    MatmulCommand *q0mm;
    MatmulCommand *k0mm;
    MatmulCommand *v0mm;
    MatmulCommand *wo0mm;
    RowMatmulSlice* q0Slice;
    RowMatmulSlice* k0Slice;
    RowMatmulSlice* v0Slice;
    ColMatmulSlice* wo0Slice;

    MatmulCommand *w10mm;
    MatmulCommand *w20mm;
    MatmulCommand *w30mm;
    RowMatmulSlice* w10Slice;
    ColMatmulSlice* w20Slice;
    RowMatmulSlice* w30Slice;

    MatmulCommand* moeRouterMm;
    RowMatmulSlice* moeUpAndGate0Slice;
    RowMatmulSlice* moeDown0Slice;
    MatmulCommand** moeUpMm;
    MatmulCommand** moeGateMm;
    MatmulCommand** moeDownMm;

    float* moeRouterProbs;
    float* expertGate;
    float* expertDown;
    float* hb20;

    KvCacheSlice* kvCacheSlice;
    float* keyCache;
    float* valueCache;
    MultiHeadAttSlice* multiHeadAttSlice;
    float* att;
    float* qo0;

    TransformerBlock(TransformerSpec* spec, slice_index_t sliceIndex);
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
    void** buffers;
    size_t* bufferBytes;

    TransformerBuffer(TransformerSpec* spec);
    ~TransformerBuffer();
    void* getUnit(uint8_t bufferIndex);
    size_t getUnitBytes(uint8_t bufferIndex);
    void* getSliced(uint8_t bufferIndex, slice_index_t sliceIndex, int midSumMemoryBudget, int sumMemoryBudget, bool IsRow);
    size_t getSlicedBytes(uint8_t bufferIndex, int sumMemoryBudget, bool IsRow);
};

class Transformer { // --> 完整的Transformer结构,包括Spec,Block,Buffer等的定义
public:
    TransformerSpec* spec;
    TransformerBlock** blocks; // --> 通过nLayers进行循环赋予空间
    TransformerBuffer* buffer;
    slice_index_t sliceIndex;

    size_t tokenEmbeddingTableBytes;
    float* tokenEmbeddingTable;
    size_t rmsFinalBytes;
    float* rmsFinal;
    MatmulCommand* wclsMm;

    pos_t pos;
    float rms;
    float* x;
    float* logits;
    RopeSlice* ropeSlice;
    RopeCommand* rope;

    ~Transformer();

    static TransformerSpec loadSpecFromFile(const char* path, const unsigned int nSlices, FloatType weightsFloatType, FloatType bufferFloatType, std::vector<int> memoryBudgetArray); // 从一个file中加载Transformer需要的基本配置信息
    static Transformer loadRootFromFile(const char* path, TransformerSpec* spec, SocketPool* socketPool); // --> 从file直接加载Transformer的完整结构
    static Transformer loadRoot(char* data, TransformerSpec* spec, SocketPool* socketPool); // --> 针对ROOT Node的加载,主要从文件中读取权重,并分发给Worker Node
    static Transformer loadSlice(TransformerSpec* spec, Socket* socket, std::vector<int> memoryBudgetArray); // --> 针对Worker Node的加载,主要从Socket中读取权重矩阵等

private:
    Transformer(TransformerSpec* spec, slice_index_t sliceIndex); // --> Transformer函数
};

#endif
