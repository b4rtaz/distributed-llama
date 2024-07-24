#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <cstdio>
#include "quants.hpp"

// RESPONSIBILITIES
//
// *Slice - calculates sizes, offsets, slice sizes etc. It is not responsible for memory allocation. It may help in the loading of data.
// *Command - allocates memory for weights, performs calculations.

typedef unsigned int pos_t;
typedef uint8_t slice_index_t;

class MatmulSlice {
public:
    size_t bytes;
    size_t sliceBytes;
    virtual size_t splitWeights(slice_index_t sliceIndex, char* weights, char* weights0) = 0;
};

class RowMatmulSlice : public MatmulSlice {
public:
    FloatType type;
    int nSlices;
    int n;
    int d0;

    RowMatmulSlice(FloatType type, int nSlices, int n, int d);
    size_t splitWeights(slice_index_t sliceIndex, char* weights, char* weights0);
    unsigned int dOffset(slice_index_t sliceIndex);
};

class ColMatmulSlice : public MatmulSlice {
public:
    FloatType type;
    int nSlices;
    int n;
    int n0;
    int d;

    ColMatmulSlice(FloatType type, int nSlices, int n, int d);
    size_t splitWeights(slice_index_t sliceIndex, char* weights, char* weights0);
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
    RopeSlice(unsigned int dim, unsigned int kvDim, unsigned int nKvHeads, unsigned int nSlices, unsigned int seqLen, unsigned int headSize, float ropeTheta, slice_index_t sliceIndex);
};

class KvCacheSlice {
public:
    unsigned int kvDim0;
    size_t keyCacheSize;
    size_t valueCacheSize;
    KvCacheSlice(unsigned int kvDim, unsigned int seqLen, unsigned int nSlices);
};

class MultiHeadAttSlice {
public:
    unsigned int nHeads0;
    size_t attSize;
    MultiHeadAttSlice(unsigned int nHeads, unsigned int seqLen, unsigned int nSlices, slice_index_t sliceIndex);
};

class MatmulCommand {
private:
    FloatType inputFloatType;
    FloatType weightsFloatType;
    unsigned int n;
    unsigned int d;
    size_t cpuSize;
    void* cpuWeights;
public:
    MatmulCommand(const unsigned int n, const unsigned int d, const FloatType inputFloatType, const FloatType weightsFloatType);
    ~MatmulCommand();
    size_t loadWeights(const void* source);
    void forward(const void* input, float* output, const unsigned int nThreads, const unsigned int threadIndex);
};

class RopeCommand {
public:
    virtual ~RopeCommand() {};
    virtual void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex) = 0;
};

class LlamaRopeCommand : public RopeCommand {
private:
    RopeSlice* slice;
    float* cache;
public:
    LlamaRopeCommand(RopeSlice *slice);
    ~LlamaRopeCommand();
    void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex);
};

class Llama3_1RopeCommand : public RopeCommand {
private:
    RopeSlice* slice;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    int ropeScalingOrigMaxSeqLen;
public:
    Llama3_1RopeCommand(RopeSlice *slice, float ropeScalingFactor, float ropeScalingLowFreqFactor, float ropeScalingHighFreqFactory, int ropeScalingOrigMaxSeqLen);
    void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex);
    float scale(float freq);
};

class FalconRopeCommand : public RopeCommand {
private:
    RopeSlice* slice;
public:
    FalconRopeCommand(RopeSlice *slice);
    ~FalconRopeCommand();
    void forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex);
};

#endif
