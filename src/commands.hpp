#ifndef COMMANDS_HPP
#define COMMANDS_HPP


#include <cstdio>
#include "quants.hpp"
// 定义了不同量化下的不同类

// RESPONSIBILITIES
//
// *Slice - calculates sizes, offsets, slice sizes etc. It is not responsible for memory allocation. It may help in the loading of data.
// *Command - allocates memory for weights, performs calculations.
// Command用来分配权重?

typedef unsigned short pos_t; // pos_t --> unsigned short
typedef uint8_t slice_index_t; // slice_index_t --> uint8_t

class MatmulSlice {
public:
    size_t bytes;
    size_t sliceBytes; // 这里已经考虑了d0 * n
    virtual size_t splitWeights(slice_index_t sliceIndex, char* weights, char* weights0, int memoryBudget, int midSumMemoryBudget) = 0;
    // class A和class B均继承自MatmulSlice，可以对class A中的splitWeights和class B中的splitWeights分别进行不同的定义
    // =0 表示这是一个纯虚函数
};
// 矩阵乘法切片

class RowMatmulSlice : public MatmulSlice { // RowMatmulSlice --> 行切分,对矩阵进行Row方向的切分
public:
    FloatType type;
    int nSlices;
    int n;
    int d0;
    int memoryBudget;
    // int d1; 调试使用的

    RowMatmulSlice(FloatType type, int nSlices, int n, int d,  int memoryBudget, int sumMemoryBudget); // --> 行切分
    size_t splitWeights(slice_index_t sliceIndex, char* weights, char* weights0, int memoryBudget, int midSumMemoryBudget);
    unsigned int dOffset(slice_index_t sliceIndex, int sumMemoryBudget); // --> 偏移值
};


class ColMatmulSlice : public MatmulSlice { // --> 列切分
public:
    FloatType type;
    int nSlices;
    int n;
    int n0;
    int d;
    int memoryBudget;

    ColMatmulSlice(FloatType type, int nSlices, int n, int d, int memoryBudget, int sumMemoryBudget); // --> 列切分
    size_t splitWeights(slice_index_t sliceIndex, char* weights, char* weights0, int memoryBudget, int midSumMemoryBudget);
};

class RopeSlice {
public:
    unsigned int qDim0;
    unsigned int qDimStart;
    unsigned int qDimEnd;
    unsigned int qShift;
    unsigned int kvDim;
    unsigned int kvDim0;
    unsigned int kvDimStart;
    unsigned int sliceDim;
    unsigned int seqLen;
    unsigned int headSize;
    unsigned int nKvHeads;
    float ropeTheta;
    RopeSlice(unsigned int dim, unsigned int kvDim, unsigned int nKvHeads, unsigned int nSlices, unsigned int seqLen, unsigned int headSize, float ropeTheta, slice_index_t sliceIndex, int memoryBudget, int midSumMemoryBudget, int sumMemoryBudget);
};

class KvCacheSlice { // --> 对KV-Cache进行切片
public:
    unsigned int kvDim0;
    size_t keyCacheSize;
    size_t valueCacheSize;
    KvCacheSlice(unsigned int kvDim, unsigned int seqLen, unsigned int nSlices, int memoryBudget, int sumMemoryBudget);
};

class MultiHeadAttSlice {
public:
    unsigned int nHeads0;
    size_t attSize;
    MultiHeadAttSlice(unsigned int nHeads, unsigned int seqLen, unsigned int nSlices, slice_index_t sliceIndex, int memoryBudget, int sumMemoryBudget);
};
// 对MHA部分进行切片

class MatmulCommand { // --> 
private:
    FloatType inputFloatType;
    FloatType weightsFloatType;
    // unsigned int n;
    // unsigned int d;
    int memoryBudget;
    size_t cpuSize; // --> cpuWeights占用的内存大小
    void* cpuWeights; // --> 指向存储权重数据的指针
public:
    unsigned int n;
    unsigned int d;
    MatmulCommand(const unsigned int n, const unsigned int d, const FloatType inputFloatType, const FloatType weightsFloatType, int memoryBudget, int FLAG);
    ~MatmulCommand();
    size_t loadWeights(const void* source); // --> 将source的内容复制到cpu中
    void forward(const void* input, float* output, const unsigned int nThreads, const unsigned int threadIndex, int FLAG); // --> 执行矩阵的前向传播操作
};

class RopeCommand {
public:
    virtual ~RopeCommand() {};
    virtual void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex, int memoryBudget) = 0; // --> 纯虚函数,应该在定义中根据isQ的值进行判断后进行不同的操作
};

class LlamaRopeCommand : public RopeCommand {
private:
    RopeSlice* slice;
    float* cache;
public:
    LlamaRopeCommand(RopeSlice *slice);
    ~LlamaRopeCommand();
    void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex, int memoryBudget);
};

class FalconRopeCommand : public RopeCommand {
private:
    RopeSlice* slice;
public:
    FalconRopeCommand(RopeSlice *slice);
    ~FalconRopeCommand();
    void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex, int memoryBudget);
};

#endif
