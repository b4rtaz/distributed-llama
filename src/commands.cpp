#include <cassert>
#include <cstring>
#include <cmath>
#include "utils.hpp"
#include "funcs.hpp"
#include "commands.hpp"

RowMatmulSlice::RowMatmulSlice(FloatType type, int nSlices, int n, int d, int memoryBudget, int sumMemoryBudget) {
    assert(d % nSlices == 0);
    assert(d % sumMemoryBudget == 0);
    this->type = type;
    this->nSlices = nSlices;
    // this->d0 = d / nSlices; // --> 这里的d指的是总维度,d0指的是每个Slice的维度
    // printf("RowMatmulSlice --> d0 %d\n", this->d0);
    // 这里很明显是按照slices的数量均分的
    this->memoryBudget = memoryBudget;
    this->d0 = d / sumMemoryBudget;
    // printf("RowMatmulSlice --> d1 %d\n", this->d1);
    this->n = n;
    this->bytes = getBatchBytes(type, this->n, d); // --> 占用总内存
    this->sliceBytes = getBatchBytes(type, this->n, this->d0); // --> 每个Slice占用的内存
}

size_t RowMatmulSlice::splitWeights(slice_index_t sliceIndex, char* weights, char* weights0, int memoryBudget, int midSumMemoryBudget) { // -->总之就是把weights中的权重参数拷贝到weights0中
    // weights --> w
    int numbersPerBatch = getNumbersPerBatch(this->type); // numbersPerBatch == 32
    int batchBytes = getBatchBytes(this->type, numbersPerBatch, 1); // 32 * 1 * 量化方式内存

    int n = this->n / numbersPerBatch;
    // this->n: dim, this->d0: dim of every slice

    // 这里的offset是当前这个切片的起始位置的偏移
    // 修改后的起始位置应该是: 
    size_t offset = this->d0 * midSumMemoryBudget * n * batchBytes; // this->d0 * sliceIndex * this->n * 量化方式内存
    size_t copiedBytes = 0;
    // this->d0 & n 就相当于一个二维矩阵的两个维度
    for (int d = 0; d < this->d0 * memoryBudget; d++) {
        for (int j = 0; j < n; j++) {
            long o = (d * n + j) * batchBytes; // --> 偏移量o

            memcpy(weights0 + o, weights + offset + o, batchBytes); // 赋值数据
            // printf("\n POS 6 %d\n", batchBytes);
            copiedBytes += batchBytes;
        }
    }
    return copiedBytes; // 拷贝总的字节数
}

unsigned int RowMatmulSlice::dOffset(slice_index_t sliceIndex, int midSumMemoryBudget) {
    return this->d0 * midSumMemoryBudget;
}

ColMatmulSlice::ColMatmulSlice(FloatType type, int nSlices, int n, int d, int memoryBudget, int sumMemoryBudget) {
    assert(n % nSlices == 0);
    this->memoryBudget = memoryBudget;
    this->type = type;
    this->nSlices = nSlices;
    this->n = n;
    // this->n0 = n / nSlices;
    this->n0 = n / sumMemoryBudget;
    this->d = d;
    this->bytes = getBatchBytes(type, n, d);
    this->sliceBytes = getBatchBytes(type, this->n0, d);// 这里的sliceBytes是单片，没有考虑memoryBudget的情况
}

size_t ColMatmulSlice::splitWeights(slice_index_t sliceIndex, char* weights, char* weights0, int memoryBudget, int midSumMemoryBudget) {
    int numbersPerBatch = getNumbersPerBatch(this->type);
    int batchBytes = getBatchBytes(this->type, numbersPerBatch, 1);
    int n0 = this->n0;
    assert(n0 % numbersPerBatch == 0);

    int n = this->n / numbersPerBatch;
    int rowBytes = n * batchBytes;
    int row0Bytes = (n0 / numbersPerBatch) * batchBytes * memoryBudget;
    int rowOffsetBytes = row0Bytes / memoryBudget * midSumMemoryBudget;
    size_t copiedBytes = 0;
    for (int d = 0; d < this->d; d++) {
        memcpy(&weights0[row0Bytes * d], &weights[rowBytes * d + rowOffsetBytes], row0Bytes);
        copiedBytes += row0Bytes;
    }

    return copiedBytes;


}

RopeSlice::RopeSlice(unsigned int dim, unsigned int kvDim, unsigned int nKvHeads, unsigned int nSlices, unsigned int seqLen, unsigned int headSize, float ropeTheta, slice_index_t sliceIndex, int memoryBudget, int midSumMemoryBudget, int sumMemoryBudget) {
    // printf("RopeSlice -> dim -> %d", dim);
    // printf("RopeSlice -> kvdim -> %d", kvDim);

    assert(dim >= kvDim);
    assert(dim % nSlices == 0);
    assert(kvDim % nSlices == 0);
    assert(dim % sumMemoryBudget == 0);
    assert(kvDim % sumMemoryBudget == 0);

    // qDim0 = dim / nSlices;
    qDim0 = dim / sumMemoryBudget;
    // kvDim0 = kvDim / nSlices;
    kvDim0 = kvDim / sumMemoryBudget;
    assert(qDim0 % 2 == 0);
    assert(kvDim0 % 2 == 0);
    kvDimStart = kvDim0 * midSumMemoryBudget;
    qDimStart = qDim0 * midSumMemoryBudget;
    qDimEnd = qDimStart + qDim0 * memoryBudget;
    qShift = qDimStart - kvDimStart; // 查询起始位置与键值起始位置之间的偏移量
    sliceDim = qDimEnd - kvDimStart;
    this->kvDim = kvDim;
    this->nKvHeads = nKvHeads;
    this->seqLen = seqLen;
    this->headSize = headSize;
    this->ropeTheta = ropeTheta;
    assert(sliceDim % 2 == 0);
}

KvCacheSlice::KvCacheSlice(unsigned int kvDim, unsigned int seqLen, unsigned int nSlices, int memoryBudget, int sumMemoryBudget) {
    assert(kvDim % nSlices == 0); // 保证KvDim是nSlices的整数倍,这样保证每个Slice上分配的KvDim0是整数
    assert(kvDim % sumMemoryBudget == 0); // 保证KvDim是sumMemoryBudget的整数倍,保证每个设备上分的KvDim0是一个整数
    // kvDim0 = kvDim / nSlices;
    // printf("kvDim0 before %d\n", kvDim0);

    kvDim0 = kvDim / sumMemoryBudget;
    // printf("kvDim0 after %d\n", kvDim0);
    keyCacheSize = seqLen * kvDim0 * sizeof(float); // --> 求key的size, 序列最大 * 每个Slice分配到的Kv维度 * 一个Kv的size也就是一个float, 保证key的缓存空间足够
    valueCacheSize = seqLen * kvDim0 * sizeof(float); // --> 同上
}

MultiHeadAttSlice::MultiHeadAttSlice(unsigned int nHeads, unsigned int seqLen, unsigned int nSlices, slice_index_t sliceIndex, int memoryBudget, int sumMemoryBudget) {
    // printf("ttttttttt %d\n", nHeads);
    assert(nHeads % nSlices == 0); // 保证Heads个数是nSlices的整数倍
    assert(nHeads % sumMemoryBudget == 0);
    // nHeads0 = nHeads / nSlices;
    nHeads0 = nHeads / sumMemoryBudget * memoryBudget;
    attSize = seqLen * nHeads0 * sizeof(float);
}

MatmulCommand::MatmulCommand(const unsigned int n, const unsigned int d, const FloatType inputFloatType, const FloatType weightsFloatType, int memoryBudget, int FLAG) {
    // --> n, dim;d, dim for every slice
    this->inputFloatType = inputFloatType;
    this->weightsFloatType = weightsFloatType;
    this->memoryBudget = memoryBudget;
    // this->n = n;
    // this->d = d;
    if(FLAG == 0){
        this->n = n;
        this->d = d * memoryBudget;
        this->cpuSize = getBatchBytes(weightsFloatType, n, d * memoryBudget); // 横切
    } else if (FLAG == 1) {
        this->n = n * memoryBudget;
        this->d = d;
        this->cpuSize = getBatchBytes(weightsFloatType, n * memoryBudget, d); // 纵切
    } else {
        this->n = n;
        this->d = d;
        this->cpuSize = getBatchBytes(weightsFloatType, n, d);
    }
    this->cpuWeights = newBuffer(this->cpuSize); // 这里的cpuWeights是 * memoryBudget之后的结果
};

MatmulCommand::~MatmulCommand() {
    freeBuffer(cpuWeights);
}

size_t MatmulCommand::loadWeights(const void* source) { // --> 将source的内容复制到cpu中
    memcpy(cpuWeights, source, cpuSize);
    // printf("\n cpuSize %ld\n", cpuSize);
    return cpuSize;
}

void MatmulCommand::forward(const void* input, float* output, const unsigned int nThreads, const unsigned int threadIndex, int FLAG) {
    // 这里的n和d都是已经 * memoryBugdet之后的结果
    // n按比例分配
    // if (FLAG == 1){
    //     printf("\n Here n:%d \t d:%d \n");
    // }
    matmul(weightsFloatType, inputFloatType, output, input, cpuWeights, n, d, nThreads, threadIndex, memoryBudget, FLAG);
    // cpuWeights是 * memoryBudget之后的结果
}

LlamaRopeCommand::LlamaRopeCommand(RopeSlice *slice) {
    this->slice = slice;

    size_t cacheBytes = slice->seqLen * slice->sliceDim * sizeof(float);
    cache = (float*)newBuffer(cacheBytes);
    printf("🕒 ropeCache: %ld kB\n", cacheBytes / 1024);

    for (pos_t pos = 0; pos < slice->seqLen; pos++) {
        for (unsigned int i = slice->kvDimStart; i < slice->qDimEnd; i += 2) {
            const unsigned int headDim = i % slice->headSize;
            const float freq = 1.0f / powf(slice->ropeTheta, headDim / (float)slice->headSize);
            const float val = pos * freq;
            const float fcr = cosf(val);
            const float fci = sinf(val);
            cache[pos * slice->sliceDim + (i - slice->kvDimStart)] = fcr;
            cache[pos * slice->sliceDim + (i - slice->kvDimStart) + 1] = fci;
        }
    }
};

LlamaRopeCommand::~LlamaRopeCommand() {
    freeBuffer(cache);
}

void LlamaRopeCommand::forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex, int memoryBudget) {
    const unsigned int dim0Half = (isQ ? slice->qDim0 * memoryBudget: slice->kvDim0 * memoryBudget) / 2;
    const unsigned int shift = isQ ? slice->qShift : 0;
    // qShift = qDimStart - kvDimStart;
    SPLIT_RANGE_TO_THREADS(s, e, 0, dim0Half, nThreads, threadIndex);
    const unsigned int iStart = s * 2;
    const unsigned int iEnd = e * 2;
    // iStart = 0 | iEnd = dim0Half * 2

    for (unsigned int i = iStart; i < iEnd; i += 2) {
        float fcr = cache[pos * slice->sliceDim + shift + i];
        float fci = cache[pos * slice->sliceDim + shift + i + 1];
        float v0 = qOrK[i];
        float v1 = qOrK[i + 1];
        qOrK[i]   = v0 * fcr - v1 * fci;
        qOrK[i + 1] = v0 * fci + v1 * fcr;
    }
}

FalconRopeCommand::FalconRopeCommand(RopeSlice *slice) {
    this->slice = slice;
}

FalconRopeCommand::~FalconRopeCommand() {}

void FalconRopeCommand::forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex, int memoryBudget) {
    // TODO: this implementation allows only a small number of slices (because it requires dim0 % headSize == 0). This could be improved.
    unsigned int dimStart = isQ ? slice->qDimStart : slice->kvDimStart;
    unsigned int dim0 = isQ ? slice->qDim0 : slice->kvDim0;
    unsigned int headSize = isQ ? slice->headSize : slice->kvDim / slice->nKvHeads;
    assert(dimStart % headSize == 0);
    assert(dim0 % headSize == 0);
    unsigned int nHeads0 = dim0 / headSize;
    SPLIT_RANGE_TO_THREADS(h0s, h0e, 0, nHeads0, nThreads, threadIndex);

    for (unsigned int h = h0s; h < h0e; h++) {
        for (unsigned int j = 0; j < headSize / 2; j++) {
            float freq = 1.0f / powf(slice->ropeTheta, 2.0f * (float)j / (float)headSize);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float q0 = qOrK[h * headSize + j];
            float q1 = qOrK[h * headSize + j + headSize / 2];
            qOrK[h * headSize + j] = q0 * fcr - q1 * fci;
            qOrK[h * headSize + j + headSize / 2] = q0 * fci + q1 * fcr;
        }
    }
}