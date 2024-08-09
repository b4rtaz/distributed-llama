#include <cstdio>
// 标准输入输出库
#include <cassert>
// assert库
#include <stdexcept>
// 提供标准异常类
#include <string.h>
// 字符处理
#include "utils.hpp"
// 主要是对线程的处理
#include "socket.hpp"
// 主要定义了一些对socket的处理
#include "commands.hpp"
// 主要定义了GEMM的乘法操作和矩阵切片等操作
#include "transformer.hpp"
// 主要定义了TransformerBlock的结构

#define IS_ROOT_SLICE(sliceIndex) (sliceIndex == 0) // IF sliceIndex == 0 THEN IS_ROOT_SLICE(sliceIndex) --> True

TransformerSpec Transformer::loadSpecFromFile(const char* path, const unsigned int nSlices, FloatType weightsFloatType, FloatType bufferFloatType, std::vector<int> memoryBudgetArray) {
    // 定义loadSpecFromFile函数,返回一个包含Transformer的信息的结构体
    TransformerSpec spec;
    memset(&spec, 0, sizeof(TransformerSpec)); // 将spec的所有成员变量初始值设置为0
    spec.hiddenAct = SILU; // 选择激活函数
    spec.ropeTheta = 10000.0f; // ropeTheta是什么?
    spec.memoryBudgetArray = memoryBudgetArray;
    FILE* fd = fopen(path, "rb"); // 打开文件
    if (fd == NULL) { // 打开文件失败
        throw std::runtime_error("Cannot open model file");
    }

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1) { // 尝试读取一个int类型的值到magic中
        throw std::runtime_error("Cannot read magic value");
    }
    if (magic == 0xABCD00 || magic == 0xABCD01) { // 如果读到的是LLAMA | GROK1, TransformerArchType
        TransformerFileOldHeader header;
        if (fread(&header, sizeof(header), 1, fd) != 1) { // 从文件中读取header
            throw std::runtime_error("Cannot read header");
        }
        // 如果过了IF,则header已经读取到了模型的相关信息
        spec.headerSize = sizeof(int) + sizeof(TransformerFileOldHeader); // 加了一个Magic参数?
        spec.archType = (TransformerArchType)magic; // Magic == ABCD00(LLAMA) OR Magic == ABCD01(GROK1)
        spec.dim = header.dim;
        spec.hiddenDim = header.hiddenDim;
        spec.nLayers = header.nLayers;
        spec.nHeads = header.nHeads;
        spec.nKvHeads = header.nKvHeads;
        spec.nExperts = header.nExperts;
        spec.nActiveExperts = header.nActiveExperts;
        spec.vocabSize = header.vocabSize;
        spec.seqLen = header.seqLen;
    } else if (magic == 0xA00ABCD) { // A00ABCD是什么? --> 好像是一种比较特殊的数据格式,不像LLAMA和GROK1可以直接读取参数,而是需要另一种方法.但是本质上都是为了读取参数;是不是应该是0xABCD02?
        if (fread(&spec.headerSize, sizeof(int), 1, fd) != 1) {
            throw std::runtime_error("Cannot read header size");
        }
        int buffer[spec.headerSize];
        if (fread(&buffer, spec.headerSize, 1, fd) != 1) {
            throw std::runtime_error("Cannot read header values");
        }
        // spec.headerSize --> header size
        // buffer --> Kv-Buffer?本质上也是为了获取spec的信息,但是方式不同
        int nKv = (spec.headerSize - 2 * sizeof(int)) / sizeof(int); // 计算Kv-Cache的个数?计算最大能存储的Kv-Cache的空间?

        FloatType modelWeightsFloatType = FUNK; // 和量化有关,不同的位数量化的存储空间也不同
        for (int i = 0; i < nKv; i += 2) {
            // 从buffer中取值,buffer中以键值对间隔保存
            int key = buffer[i];
            int value = buffer[i + 1];
            if (key == VERSION) spec.version = value;
            else if (key == ARCH_TYPE) spec.archType = (TransformerArchType)value;
            else if (key == DIM) spec.dim = value;
            else if (key == HIDDEN_DIM) spec.hiddenDim = value;
            else if (key == N_LAYERS) spec.nLayers = value;
            else if (key == N_HEADS) spec.nHeads = value;
            else if (key == N_KV_HEADS) spec.nKvHeads = value;
            else if (key == N_EXPERTS) spec.nExperts = value;
            else if (key == N_ACTIVE_EXPERTS) spec.nActiveExperts = value;
            else if (key == VOCAB_SIZE) spec.vocabSize = value;
            else if (key == SEQ_LEN) spec.seqLen = value;
            else if (key == HIDDEN_ACT) spec.hiddenAct = (TransformerHiddenAct)value; // --> 将value强制转化为TransformerHiddenAct
            else if (key == ROPE_THETA) spec.ropeTheta = (float)value; // --> 类型强制转化
            else if (key == WEIGHTS_FLOAT_TYPE) weightsFloatType = (FloatType)value;
            else {
                throw std::runtime_error("Unsupported header key");
            }
        }
    } else {
        throw std::runtime_error("Unsupported model file");
    }

    if (weightsFloatType == FUNK)
        throw std::runtime_error("Not specified weights float type"); // 不支持的量化格式

    spec.headSize = spec.dim / spec.nHeads;
    spec.kvDim = (spec.dim * spec.nKvHeads) / spec.nHeads; // --> ?
    spec.weightsFloatType = weightsFloatType;
    spec.bufferFloatType = bufferFloatType; // 这个是调用函数要传入的参数
    spec.nSlices = nSlices; // --> 输入

    if (spec.nSlices > spec.nKvHeads) {
        // TODO: https://github.com/b4rtaz/distributed-llama/issues/70
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model.");
    }
    // -----------------------------------------------------
    #if 0
    printf("💡 nSlices:\n", spec.nSlices);
    printf("💡 nKvHeads:\n", spec.nKvHeads);
    printf("🫵 nSlices must less than nKvHeads!!!!!")
    #endif
    // -----------------------------------------------------
    if (spec.archType == LLAMA) {
        printf("💡 arch: llama\n");
    } else if (spec.archType == GROK1) {
        printf("💡 arch: grok1\n");
    } else if (spec.archType == MIXTRAL) {
        printf("💡 arch: mixtral\n");
    } else {
        throw std::runtime_error("Unsupported architecture");
    }
    if (spec.hiddenAct == GELU) {
        printf("💡 hiddenAct: gelu\n");
    } else if (spec.hiddenAct == SILU) {
        printf("💡 hiddenAct: silu\n");
    } else {
        throw std::runtime_error("Unsupported hidden activation");
    }
    printf("💡 dim: %d\n", spec.dim);
    printf("💡 hiddenDim: %d\n", spec.hiddenDim);
    printf("💡 nLayers: %d\n", spec.nLayers);
    printf("💡 nHeads: %d\n", spec.nHeads);
    printf("💡 nKvHeads: %d\n", spec.nKvHeads);
    if (spec.nExperts > 0) {
        printf("💡 nExperts: %d\n", spec.nExperts);
        printf("💡 nActiveExperts: %d\n", spec.nActiveExperts);
    }
    printf("💡 vocabSize: %d\n", spec.vocabSize);
    printf("💡 seqLen: %d\n", spec.seqLen);
    printf("💡 nSlices: %d\n", spec.nSlices);
    printf("💡 ropeTheta: %.1f\n", spec.ropeTheta);

    // for (int value : spec.memoryBudgetArray) {
    //     printf("💡 arg test ----------------: %d\n", value);
    // }

    spec.fileSize = (size_t)seekToEnd(fd);
    fclose(fd);
    return spec;
}
// --- 
TransformerBuffer::TransformerBuffer(TransformerSpec* spec) { // 分配内存缓冲区
    // TransformerSpec spec,使用spec.A
    // TransformerSpec* spec,使用spec->A
    nSlices = spec->nSlices;
    buffers = new void*[TB_LENGTH]; // 定义一个buffer
    bufferBytes = new size_t[TB_LENGTH];

    bufferBytes[TB_UNIT_XB] = spec->dim * sizeof(float); // --> dim是一个float,保存输入所需要占用的空间
    bufferBytes[TB_UNIT_XB_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, 1); // --> 量化后dim占用的空间
    bufferBytes[TB_SLICED_XB2] = spec->dim * sizeof(float);
    bufferBytes[TB_SLICED_XB2_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, 1);
    bufferBytes[TB_SLICED_XBV] = spec->dim * spec->nSlices * sizeof(float); // -->存储分片后的输入数据及其量化后的大小
    bufferBytes[TB_SLICED_XBV_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, spec->nSlices);
    // spec->dim * spec->nSlices * bufferFloatType

    int nHb = (spec->nActiveExperts > 0)
        ? spec->hiddenDim * spec->nActiveExperts
        : spec->hiddenDim;
    bufferBytes[TB_SLICED_HB] = nHb * sizeof(float); // -->用于存储隐藏层数据及其量化后的大小
    bufferBytes[TB_SLICED_HB_QUANTIZED] = getBatchBytes(spec->bufferFloatType, nHb, 1);

    if (spec->nActiveExperts > 0) { // -->如果有专家,则额外分配缓存区和INDEX
        bufferBytes[TB_UNIT_MOE_INDEXES] = spec->nActiveExperts * sizeof(uint8_t);
        bufferBytes[TB_UNIT_MOE_WEIGHTS] = spec->nActiveExperts * sizeof(float);

        buffers[TB_UNIT_MOE_INDEXES] = newBuffer(bufferBytes[TB_UNIT_MOE_INDEXES]);
        buffers[TB_UNIT_MOE_WEIGHTS] = newBuffer(bufferBytes[TB_UNIT_MOE_WEIGHTS]);
    } else {
        bufferBytes[TB_UNIT_MOE_INDEXES] = 0;
        bufferBytes[TB_UNIT_MOE_WEIGHTS] = 0;
    }

    for (int i = 0; i < TB_LENGTH - TB_NO_PAIRS; i += 2) { // --> 这里之所以- TB_NO_PAIRS,是因为在上面已经给专家用newBuffer分配了缓存,这里不需要再分配.[TB_UNIT_MOE_WEIGHTS]\[TB_UNIT_MOE_INDEXES]
        int bytes = bufferBytes[i];
        buffers[i] = newBuffer(bufferBytes[i]);
        if (spec->bufferFloatType == F32) {
            buffers[i + 1] = buffers[i]; // 相邻的两个buffer存储的是量化前后分配的空间
        } else {
            buffers[i + 1] = newBuffer(bufferBytes[i + 1]);
        }
        // 循环分配缓存空间
        // 如果spec->bufferFloatType == F32则量化前后的空间大小是一样的,可以共享相同的内存.不需要重新分配
    }
}

TransformerBuffer::~TransformerBuffer() {
    if (bufferBytes[TB_UNIT_MOE_INDEXES] > 0 && bufferBytes[TB_UNIT_MOE_WEIGHTS] > 0) {
        freeBuffer(buffers[TB_UNIT_MOE_INDEXES]);
        freeBuffer(buffers[TB_UNIT_MOE_WEIGHTS]);
    }

    for (int i = 0; i < TB_LENGTH - TB_NO_PAIRS; i += 2) {
        if (bufferBytes[i] > 0) {
            if (buffers[i] != buffers[i + 1]) {
                freeBuffer(buffers[i + 1]);
            }
            // 如果buffer[i] == buffer[i+1]的话,只释放其中一个?
            freeBuffer(buffers[i]);
        }
    }
    delete[] bufferBytes;
    delete[] buffers;
}

void* TransformerBuffer::getUnit(uint8_t bufferIndex) {
    return buffers[bufferIndex];
}

size_t TransformerBuffer::getUnitBytes(uint8_t bufferIndex) {
    return bufferBytes[bufferIndex];
}

void* TransformerBuffer::getSliced(uint8_t bufferIndex, slice_index_t sliceIndex, int midSumMemoryBudget, int sumMemoryBudget, bool IsRow) {
    size_t sliceBytes = getSlicedBytes(bufferIndex, sumMemoryBudget, IsRow);
    // printf("\n getSliced sliceBytes:%ld \t sliceIndex:%d\n", sliceBytes,sliceIndex);
    if(IsRow){
        return ((char*)buffers[bufferIndex]) + sliceBytes * midSumMemoryBudget; // --> 获取某个切片的缓冲区起始地址
    }else {
        return ((char*)buffers[bufferIndex]) + sliceBytes * sliceIndex;
    }
}

size_t TransformerBuffer::getSlicedBytes(uint8_t bufferIndex, int sumMemoryBudget, bool IsRow) { // --> 获取每一个Slice需要的内存大小
    if(IsRow){
        return bufferBytes[bufferIndex] / sumMemoryBudget;
    }else{
        return bufferBytes[bufferIndex] / nSlices;
    }
}

Transformer::Transformer(TransformerSpec* spec, slice_index_t sliceIndex) { // -->Transformer结构,主要是给blocks分配了缓存空间
    // 根节点传过来的sliceIndex == 0 | Worker Node传过来的就是各自从socket读取的sliceIndex
    this->spec = spec;
    this->sliceIndex = sliceIndex; // 这里的index应该只有一个,Root Node是0,其他Worker Node依次
    buffer = new TransformerBuffer(spec); // --> 给参数分配空间
    blocks = new TransformerBlock*[spec->nLayers]; // 有多少层就需要多少个Block
    // printf("\n POS0 \n");
    for (int i = 0; i < spec->nLayers; i++) {
        // 问题在这儿!!!
        blocks[i] = new TransformerBlock(spec, sliceIndex); // --> 给Q,K,V,W分配空间
        // 问题在这儿!!!
        // 给每个block按照sliceIndex进行切分了
    }
    int sumMemoryBudget = 0;
    int midSumMemoryBudget = 0;
    for (int value : spec->memoryBudgetArray) {
        sumMemoryBudget += value;
    }
    for (int i = 0; i < sliceIndex; i++) {
        midSumMemoryBudget += spec->memoryBudgetArray[i];
    }
    int memoryBudget = spec->memoryBudgetArray[this->sliceIndex];
    if (IS_ROOT_SLICE(sliceIndex)) { // --> 如果是ROOT分片
        tokenEmbeddingTableBytes = spec->vocabSize * spec->dim * sizeof(float); // --> 如果是ROOT分片,则需要额外的Embedding空间
        // 根切片需要的词嵌入表的内存大小，由词汇量和嵌入维度决定
        rmsFinalBytes = spec->dim * sizeof(float); // 存储rmsNORM最终层的空间
        tokenEmbeddingTable = (float*)newBuffer(tokenEmbeddingTableBytes);
        rmsFinal = (float*)newBuffer(rmsFinalBytes);

        wclsMm = new MatmulCommand(spec->dim, spec->vocabSize, F32, spec->weightsFloatType, memoryBudget, 2);

        x = (float*)newBuffer(spec->dim * sizeof(float)); // --> 存储输入的空间
        logits = (float*)newBuffer(spec->vocabSize * sizeof(float)); // --> 存储预测分数的内存
    }
    ropeSlice = new RopeSlice(spec->dim, spec->kvDim, spec->nKvHeads, spec->nSlices, spec->seqLen, spec->headSize, spec->ropeTheta, sliceIndex, memoryBudget, midSumMemoryBudget, sumMemoryBudget);
    if (spec->archType == GROK1 || spec->archType == MIXTRAL) {
        rope = new FalconRopeCommand(ropeSlice);
    } else {
        rope = new LlamaRopeCommand(ropeSlice);
    }
    TransformerBlock* b = blocks[0]; // --> b是第一个block
    assert(b->q0Slice->d0 == ropeSlice->qDim0);// spec->dim / nSlices
    assert(b->q0Slice->dOffset(sliceIndex, midSumMemoryBudget) == ropeSlice->qDimStart); // d0 * sliceIndex == q_dim_0 * sliceIndex
    assert(b->k0Slice->d0 == ropeSlice->kvDim0); // 
    assert(b->k0Slice->dOffset(sliceIndex, midSumMemoryBudget) == ropeSlice->kvDimStart); // d0_kv * sliceIndex == d0_kv * sliceIndex
    assert(b->kvCacheSlice->kvDim0 == ropeSlice->kvDim0);
}

Transformer::~Transformer() {
    delete buffer;
    for (int i = 0; i < spec->nLayers; i++) {
        delete blocks[i];
    }
    delete[] blocks;

    if (IS_ROOT_SLICE(sliceIndex)) {
        freeBuffer(tokenEmbeddingTable);
        freeBuffer(rmsFinal);
        delete wclsMm;

        freeBuffer(x);
        freeBuffer(logits);
    }

    delete ropeSlice;
    delete rope;
}

TransformerBlock::TransformerBlock(TransformerSpec* spec, slice_index_t sliceIndex) { // -->给Q,K,V,W分配空间
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    if (IS_ROOT_SLICE(sliceIndex)) { // --> 判断是不是根节点, 如果是根节点, 分配RMSNorm参数需要的空间,因为在Tensor并行中,需要根节点进行归一化等操作
        rmsAttBytes = spec->dim * sizeof(float);
        rmsFfnBytes = spec->dim * sizeof(float);
        rmsMoeBytes = spec->dim * sizeof(float);
        rmsFfn2Bytes = spec->dim * sizeof(float);
        // --> RMSNorm需要的参数,与dim有关

        rmsAtt = (float*)newBuffer(rmsAttBytes);
        rmsFfn = (float*)newBuffer(rmsFfnBytes);
        if (spec->archType == GROK1) {
            rmsMoe = (float*)newBuffer(rmsMoeBytes);
            rmsFfn2 = (float*)newBuffer(rmsFfn2Bytes);
        }
    }
    int sumMemoryBudget = 0;
    for (int value : spec->memoryBudgetArray) {
        sumMemoryBudget += value;
    }
    int memoryBudget = spec->memoryBudgetArray[this->sliceIndex];
    kvCacheSlice = new KvCacheSlice(spec->kvDim, spec->seqLen, spec->nSlices, memoryBudget, sumMemoryBudget); // 对KvCache进行切片
    keyCache = (float*)newBuffer(kvCacheSlice->keyCacheSize * memoryBudget);
    valueCache = (float*)newBuffer(kvCacheSlice->valueCacheSize * memoryBudget); // 给kv分配空间

    multiHeadAttSlice = new MultiHeadAttSlice(spec->nHeads, spec->seqLen, spec->nSlices, sliceIndex, memoryBudget, sumMemoryBudget);
    att = (float*)newBuffer(multiHeadAttSlice->attSize); // 给MHA切片分配空间

    // printf("------------------------spec dim %s\n", typeid(spec->dim).name());
    // spec -> dim = 4096 

    q0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->dim, memoryBudget, sumMemoryBudget);
    // q0Slice -> d0 = spec->dim / sumMemoryBudget
    k0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->kvDim, memoryBudget, sumMemoryBudget);
    // printf("\n%d\n", k0Slice->d0);
    v0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->kvDim, memoryBudget, sumMemoryBudget);
    // printf("\n%d\n", v0Slice->d0);
    // 这里的dim和KvDim有什么区别? --> 二维,占用的数据应该是dim_1 * dim_2
    // Q的第二维度和Kv的第二维度不同?
    wo0Slice = new ColMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->dim, memoryBudget, sumMemoryBudget);

    q0mm = new MatmulCommand(q0Slice->n, q0Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);
    // 求出来对每个切片的内存占用
    // input -> buffer;
    k0mm = new MatmulCommand(k0Slice->n, k0Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);
    v0mm = new MatmulCommand(v0Slice->n, v0Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);
    wo0mm = new MatmulCommand(wo0Slice->n0, wo0Slice->d, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 1);
    // wo0mm 传入的wo0Slice->d = spec->dim
    qo0 = (float*)newBuffer(q0Slice->d0 * memoryBudget * sizeof(float)); // --> 每个d0占用的维度

    if (spec->nExperts > 0) {
        moeUpAndGate0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim, memoryBudget, sumMemoryBudget);
        moeDown0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->hiddenDim, spec->dim, memoryBudget, sumMemoryBudget);

        moeRouterProbs = (float*)newBuffer(spec->nExperts * sizeof(float));

        moeUpMm = new MatmulCommand*[spec->nExperts];
        moeGateMm = new MatmulCommand*[spec->nExperts];
        moeDownMm = new MatmulCommand*[spec->nExperts];
        moeRouterMm = new MatmulCommand(spec->dim, spec->nExperts, F32, spec->weightsFloatType, memoryBudget, 0);

        for (int e = 0; e < spec->nExperts; e++) {
            moeUpMm[e] = new MatmulCommand(moeUpAndGate0Slice->n, moeUpAndGate0Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);
            moeGateMm[e] = new MatmulCommand(moeUpAndGate0Slice->n, moeUpAndGate0Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);
            moeDownMm[e] = new MatmulCommand(moeDown0Slice->n, moeDown0Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);
        }

        expertGate = (float*)newBuffer(moeUpAndGate0Slice->d0 * spec->nExperts * sizeof(float));
        expertDown = (float*)newBuffer(moeDown0Slice->d0 * (spec->nExperts - 1) * sizeof(float));
    } else {
        // spec->hiddenDim == 5632
        w10Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim, memoryBudget, sumMemoryBudget);
        w20Slice = new ColMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->hiddenDim, spec->dim, memoryBudget, sumMemoryBudget);
        w30Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim, memoryBudget, sumMemoryBudget);

        w10mm = new MatmulCommand(w10Slice->n, w10Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);
        w20mm = new MatmulCommand(w20Slice->n0, w20Slice->d, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 1);
        w30mm = new MatmulCommand(w30Slice->n, w30Slice->d0, spec->bufferFloatType, spec->weightsFloatType, memoryBudget, 0);

        hb20 = (float*)newBuffer(w30Slice->d0 * memoryBudget * sizeof(float));
    }
}

TransformerBlock::~TransformerBlock() {
    if (IS_ROOT_SLICE(sliceIndex)) {
        freeBuffer(rmsAtt);
        freeBuffer(rmsFfn);
        if (spec->archType == GROK1) {
            freeBuffer(rmsMoe);
            freeBuffer(rmsFfn2);
        }
    }

    delete kvCacheSlice;
    freeBuffer(keyCache);
    freeBuffer(valueCache);
    delete multiHeadAttSlice;
    freeBuffer(att);

    delete q0Slice;
    delete k0Slice;
    delete v0Slice;
    delete wo0Slice;

    freeBuffer(qo0);

    delete q0mm;
    delete k0mm;
    delete v0mm;
    delete wo0mm;

    if (spec->nExperts > 0) {
        delete moeUpAndGate0Slice;
        delete moeDown0Slice;

        for (int e = 0; e < spec->nExperts; e++) {
            delete moeUpMm[e];
            delete moeGateMm[e];
            delete moeDownMm[e];
        }

        delete[] moeUpMm;
        delete[] moeGateMm;
        delete[] moeDownMm;
        freeBuffer(moeRouterProbs);

        freeBuffer(expertGate);
        freeBuffer(expertDown);
    } else {
        delete w10Slice;
        delete w20Slice;
        delete w30Slice;

        delete w10mm;
        delete w20mm;
        delete w30mm;

        freeBuffer(hb20);
    }
}

static size_t loadSlicedMatmulWeights(const uint8_t nSlices, MatmulSlice* slice, char* source, MatmulCommand* mm, SocketPool* socketPool, std::vector<int> memoryBudgetArray) {
    // w += loadSlicedMatmulWeights(spec->nSlices, block->q0Slice, w, block->q0mm, socketPool);
    // source --> w
    // slice --> block->q0slice
    // int sumMemoryBudget = 0;
    // for (int value: memoryBudgetArray){
    //     sumMemoryBudget += value;
    // }

    size_t loadedBytes = 0;


    // nSlices == 2
    for (uint8_t s = 0; s < nSlices; s++) {
        slice_index_t sliceIndex = (s + 1) % nSlices; // --> 从这里才得到sliceIndex
        char* buffer = (char*)newBuffer(slice->sliceBytes * memoryBudgetArray[sliceIndex]); // --> 切片尺寸 --> 此时已经进行了线性分配
        // 计算不同sliceIndex占用的内存?
        // s == 0 --> sliceIndex == 1
        // 应该跟这里的d0没有关系了 --> 用sliceIndex进行分配
        // printf("\n buffer length %ld \n", slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        int midSumMemoryBudget = 0;
        for (int i = 0; i < sliceIndex; i++){
            midSumMemoryBudget += memoryBudgetArray[i];
        }
        loadedBytes += slice->splitWeights(sliceIndex, source, buffer, memoryBudgetArray[sliceIndex], midSumMemoryBudget); // -->总之就是根据sliceIndex把source中的权重参数拷贝到buffer中,并记录拷贝的字节数
        // buffer是公用的,因此一定是不断覆盖的.
        // for (int i = 0; i < 10; i++){
        //     printf("\n loadSlicedMatmulWeights buffer %c \n", buffer[i]);
        // }
        if (sliceIndex > 0) { // 对于非Root Node
            unsigned int socketIndex = sliceIndex - 1;
            // printf("\n 发送数据 %ld \n", slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            // for (int i = 0; i < 100; i++){
            //     printf("\n loadSlicedMatmulWeights buffer %c \n", buffer[i]);
            // }
            socketPool->write(socketIndex, buffer, slice->sliceBytes * memoryBudgetArray[sliceIndex]); // -->把刚才从内存中读取的slice权重数据发送给对应的Node
        } else {
            // printf("\n 加载数据 %ld \n", slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            // for (int i = 0; i < 100; i++){
            //     printf("\n loadSlicedMatmulWeights buffer %c \n", buffer[i]);
            // }
            mm->loadWeights(buffer); // 将buffer的内容复制到cpu中
            // --> 如果是ROOT节点则直接load Weight即可,如果不是ROOT节点则需要将权重分发
        }
        freeBuffer(buffer);
    }

    return loadedBytes;
}

static size_t loadRootWeights(char** target, char* source, size_t bytes) { // --> 加载根节点权重
    memcpy(*target, source, bytes);
    return bytes;
}

static size_t readSlicedMatmulWeights(MatmulSlice* slice, char* weights0, Socket* socket) { // --> 
    socket->read(weights0, slice->sliceBytes); // --> 这里应该是对于Worker Node而言,从socket中读取权重数据.
    return slice->sliceBytes;
}

Transformer Transformer::loadRootFromFile(const char* path, TransformerSpec* spec, SocketPool* socketPool) {
    MmapFile file;
    openMmapFile(&file, path, spec->fileSize);

    char* weights = ((char*)file.data) + spec->headerSize; // --> 加载权重
    Transformer transformer = Transformer::loadRoot((char*)weights, spec, socketPool);

    closeMmapFile(&file);

    return transformer;
}

Transformer Transformer::loadRoot(char* data, TransformerSpec* spec, SocketPool* socketPool) {
    assert(socketPool->nSockets == spec->nSlices - 1); // --> Sockets的数量不包含ROOT节点

    const slice_index_t sliceIndex = 0; // Root slice
    Transformer transformer(spec, sliceIndex); // --> 根据spec和sliceIndex进行内存分配, 如果sliceIndex == 0则需要分配额外空间.
    if (spec->nSlices > 1) { // --> 如果需要使用Distributed
        for (slice_index_t sliceIndex = 1; sliceIndex < spec->nSlices; sliceIndex++) {
            unsigned int socketIndex = sliceIndex - 1;
            socketPool->write(socketIndex, (char*)&sliceIndex, sizeof(uint8_t));
            socketPool->write(socketIndex, (char*)spec, sizeof(TransformerSpec));
            // --> ROOT依次给每个Worker Node分配sliceIndex 和 spec
        }
    }
    // 第一个Worker Node拿到的sliceIndex == 1

    char* w = data;// --> 这里的w是一个字符指针,w的位置始终指向下一个需要加载的位置,通过移动w的位置来加载权重
    // printf("\n START POS %p \n", w);
    w += loadRootWeights((char**)&transformer.tokenEmbeddingTable, w, transformer.tokenEmbeddingTableBytes);
    // printf("\n loadRootWeights POS %p \n", w);
    // 把w指针位置的大小为transformer.tokenEmbeddingTableBytes的权重加载到transformer.tokenEmbeddingTable中,同时将w后移.
    // spec->nLayers

    char* mid = w;
    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];
        w += loadSlicedMatmulWeights(spec->nSlices, block->q0Slice, w, block->q0mm, socketPool, spec->memoryBudgetArray);
        w += loadSlicedMatmulWeights(spec->nSlices, block->k0Slice, w, block->k0mm, socketPool, spec->memoryBudgetArray);
        w += loadSlicedMatmulWeights(spec->nSlices, block->v0Slice, w, block->v0mm, socketPool, spec->memoryBudgetArray);
        // printf("\n START wo0mm \n");
        w += loadSlicedMatmulWeights(spec->nSlices, block->wo0Slice, w, block->wo0mm, socketPool, spec->memoryBudgetArray);
        // printf("\n END wo0mm \n");

        /*
        分别加载q,k,v,wo矩阵,除了ROOT节点直接加载,其他节点的权重使用socket发送.在加载权重的过程中,不断移动w
        其中ROOT节点的权重加载到block->q0mm等.
        */
        if (spec->nExperts > 0) {
            w += block->moeRouterMm->loadWeights(w);

            for (int e = 0; e < spec->nExperts; e++) {
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeUpAndGate0Slice, w, block->moeUpMm[e], socketPool, spec->memoryBudgetArray);
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeUpAndGate0Slice, w, block->moeGateMm[e], socketPool, spec->memoryBudgetArray);
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeDown0Slice, w, block->moeDownMm[e], socketPool, spec->memoryBudgetArray);
            }
        } else {
            w += loadSlicedMatmulWeights(spec->nSlices, block->w10Slice, w, block->w10mm, socketPool, spec->memoryBudgetArray);
            // printf("\n START w20mm \n");
            w += loadSlicedMatmulWeights(spec->nSlices, block->w20Slice, w, block->w20mm, socketPool, spec->memoryBudgetArray);
            // printf("\n END w20mm \n");
            w += loadSlicedMatmulWeights(spec->nSlices, block->w30Slice, w, block->w30mm, socketPool, spec->memoryBudgetArray);
        }
        // printf("\n w30mm - mid %ld\n", (long)(w - mid));
        w += loadRootWeights((char**)&block->rmsAtt, w, block->rmsAttBytes);
        // printf("\n rmsAtt - mid %ld\n", (long)(w - mid));
        w += loadRootWeights((char**)&block->rmsFfn, w, block->rmsFfnBytes);
        // printf("\n rmsFfn - mid %ld\n", (long)(w - mid));


        if (spec->archType == GROK1) {
            w += loadRootWeights((char**)&block->rmsMoe, w, block->rmsMoeBytes);
            w += loadRootWeights((char**)&block->rmsFfn2, w, block->rmsFfn2Bytes);
        }
    }
    w += loadRootWeights((char**)&transformer.rmsFinal, w, transformer.rmsFinalBytes);
    // printf("\n rmsFinal - mid %ld\n", (long)(w - mid));
    w += transformer.wclsMm->loadWeights(w);
    // printf("\n wclsMm - mid %ld\n", (long)(w - mid));
    long missedBytes = (long)(w - data) - spec->fileSize + spec->headerSize; // --> 计算是否存在文件不完整
    if (missedBytes != 0) {
        printf("The model file is missing %ld bytes\n", missedBytes);
        exit(EXIT_FAILURE);
    }

    printf("⏩ Loaded %ld kB\n", (long)(w - data) / 1024);
    return transformer;
}

Transformer Transformer::loadSlice(TransformerSpec* spec, Socket* socket, std::vector<int> memoryBudgetArray) {
    // Worker Node从socket中读取自己的数据
    slice_index_t sliceIndex;
    socket->read((char*)&sliceIndex, sizeof(uint8_t));
    socket->read((char*)spec, sizeof(TransformerSpec));
    spec->memoryBudgetArray = memoryBudgetArray;
    // --> 从socket中读取自己的Index和spec
    // printf("555555555555555555");
    printf("💡 sliceIndex: %d\n", sliceIndex);
    printf("💡 nSlices: %d\n", spec->nSlices);
    // printf("555555555555555555");

    assert(sliceIndex >= 1); 
    Transformer transformer(spec, sliceIndex);

    size_t bufferSize = 0;
    // TODO: this is ugly
    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];
        if (block->k0Slice->sliceBytes * memoryBudgetArray[sliceIndex] > bufferSize) bufferSize = block->k0Slice->sliceBytes * memoryBudgetArray[sliceIndex];
        if (block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex] > bufferSize) bufferSize = block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex];
        if (block->wo0Slice->sliceBytes * memoryBudgetArray[sliceIndex] > bufferSize) bufferSize = block->wo0Slice->sliceBytes * memoryBudgetArray[sliceIndex];
        if (spec->nExperts > 0) {
            printf("\n HAVE EXPERTS \n");
            if (block->moeUpAndGate0Slice[0].sliceBytes > bufferSize) bufferSize = block->moeUpAndGate0Slice[0].sliceBytes;
            if (block->moeDown0Slice[0].sliceBytes > bufferSize) bufferSize = block->moeDown0Slice[0].sliceBytes;
        } else {
            if (block->w10Slice->sliceBytes * memoryBudgetArray[sliceIndex] > bufferSize) bufferSize = block->w10Slice->sliceBytes * memoryBudgetArray[sliceIndex];
            if (block->w20Slice->sliceBytes * memoryBudgetArray[sliceIndex] > bufferSize) bufferSize = block->w20Slice->sliceBytes * memoryBudgetArray[sliceIndex];
            if (block->w30Slice->sliceBytes * memoryBudgetArray[sliceIndex] > bufferSize) bufferSize = block->w30Slice->sliceBytes * memoryBudgetArray[sliceIndex];
        }
    }
    // 计算一个Block所需要的最大的缓冲区大小

    char* buffer = new char[bufferSize];
    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];
        size_t blockBytes = 0;
        long t0 = timeMs();

        // printf("\n read size %ld \n", block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        socket->read(buffer, block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        blockBytes += block->q0mm->loadWeights(buffer);

        // printf("\n read size %ld \n", block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        socket->read(buffer, block->k0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        blockBytes += block->k0mm->loadWeights(buffer);

        // printf("\n read size %ld \n", block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        socket->read(buffer, block->v0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        blockBytes += block->v0mm->loadWeights(buffer);

        // printf("\n read size %ld \n", block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        socket->read(buffer, block->wo0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
        blockBytes += block->wo0mm->loadWeights(buffer);

        if (spec->nExperts > 0) {
            for (int e = 0; e < spec->nExperts; e++) {
                socket->read(buffer, block->moeUpAndGate0Slice->sliceBytes);
                blockBytes += block->moeUpMm[e]->loadWeights(buffer);

                socket->read(buffer, block->moeUpAndGate0Slice->sliceBytes);
                blockBytes += block->moeGateMm[e]->loadWeights(buffer);

                socket->read(buffer, block->moeDown0Slice->sliceBytes);
                blockBytes += block->moeDownMm[e]->loadWeights(buffer);
            }
        } else {
            // printf("\n read size %ld \n", block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            socket->read(buffer, block->w10Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            blockBytes += block->w10mm->loadWeights(buffer);

            // printf("\n read size %ld \n", block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            socket->read(buffer, block->w20Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            blockBytes += block->w20mm->loadWeights(buffer);

            // printf("\n read size %ld \n", block->q0Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            socket->read(buffer, block->w30Slice->sliceBytes * memoryBudgetArray[sliceIndex]);
            blockBytes += block->w30mm->loadWeights(buffer);
        }

        float kbs = blockBytes / (float)(timeMs() - t0);
        printf("⏩ Received %ld kB for block %d (%.0f kB/s)\n", blockBytes / 1024, i, kbs);
    }

    delete[] buffer;
    return transformer;
}
