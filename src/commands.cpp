#include <cassert>
#include <cstring>
#include <cmath>
#include "utils.hpp"
#include "funcs.hpp"
#include "commands.hpp"

RowMatmulSlice::RowMatmulSlice(FloatType type, int nSlices, int n, int d) {
    assert(d % nSlices == 0);

    this->type = type;
    this->nSlices = nSlices;
    this->d0 = d / nSlices;
    this->n = n;
    this->bytes = getBatchBytes(type, this->n, d);
    this->sliceBytes = getBatchBytes(type, this->n, this->d0);
}

size_t RowMatmulSlice::splitWeights(slice_index_t sliceIndex, char* weights, char* weights0) {
    int numbersPerBatch = getNumbersPerBatch(this->type);
    int batchBytes = getBatchBytes(this->type, numbersPerBatch, 1);

    int n = this->n / numbersPerBatch;
    size_t offset = this->d0 * sliceIndex * n * batchBytes;
    size_t copiedBytes = 0;

    for (int d = 0; d < this->d0; d++) {
        for (int j = 0; j < n; j++) {
            long o = (d * n + j) * batchBytes;

            memcpy(weights0 + o, weights + offset + o, batchBytes);
            copiedBytes += batchBytes;
        }
    }
    return copiedBytes;
}

unsigned int RowMatmulSlice::dOffset(slice_index_t sliceIndex) {
    return this->d0 * sliceIndex;
}

ColMatmulSlice::ColMatmulSlice(FloatType type, int nSlices, int n, int d) {
    assert(n % nSlices == 0);

    this->type = type;
    this->nSlices = nSlices;
    this->n = n;
    this->n0 = n / nSlices;
    this->d = d;
    this->bytes = getBatchBytes(type, n, d);
    this->sliceBytes = getBatchBytes(type, this->n0, d);
}

size_t ColMatmulSlice::splitWeights(slice_index_t sliceIndex, char* weights, char* weights0) {
    int numbersPerBatch = getNumbersPerBatch(this->type);
    int batchBytes = getBatchBytes(this->type, numbersPerBatch, 1);
    assert(n0 % numbersPerBatch == 0);

    int n = this->n / numbersPerBatch;
    int rowBytes = n * batchBytes;
    int row0Bytes = (n0 / numbersPerBatch) * batchBytes;
    int rowOffsetBytes = sliceIndex * row0Bytes;

    size_t copiedBytes = 0;
    for (int d = 0; d < this->d; d++) {
        memcpy(&weights0[row0Bytes * d], &weights[rowBytes * d + rowOffsetBytes], row0Bytes);
        copiedBytes += row0Bytes;
    }
    return copiedBytes;
}

RopeSlice::RopeSlice(unsigned int dim, unsigned int kvDim, unsigned int nKvHeads, unsigned int nSlices, unsigned int seqLen, unsigned int headSize, float ropeTheta, slice_index_t sliceIndex) {
    assert(dim >= kvDim);
    assert(dim % nSlices == 0);
    assert(kvDim % nSlices == 0);

    qDim0 = dim / nSlices;
    kvDim0 = kvDim / nSlices;
    assert(qDim0 % 2 == 0);
    assert(kvDim0 % 2 == 0);
    kvDimStart = kvDim0 * sliceIndex;
    qDimStart = qDim0 * sliceIndex;
    qDimEnd = qDimStart + qDim0;
    qShift = qDimStart - kvDimStart;
    sliceDim = qDimEnd - kvDimStart;
    this->kvDim = kvDim;
    this->nKvHeads = nKvHeads;
    this->seqLen = seqLen;
    this->headSize = headSize;
    this->ropeTheta = ropeTheta;
    assert(sliceDim % 2 == 0);
}

KvCacheSlice::KvCacheSlice(unsigned int kvDim, unsigned int seqLen, unsigned int nSlices) {
    assert(kvDim % nSlices == 0);
    kvDim0 = kvDim / nSlices;
    keyCacheSize = seqLen * kvDim0 * sizeof(float);
    valueCacheSize = seqLen * kvDim0 * sizeof(float);
}

MultiHeadAttSlice::MultiHeadAttSlice(unsigned int nHeads, unsigned int seqLen, unsigned int nSlices, slice_index_t sliceIndex) {
    assert(nHeads % nSlices == 0);
    nHeads0 = nHeads / nSlices;
    attSize = seqLen * nHeads0 * sizeof(float);
}

AcceleratorContext::AcceleratorContext(unsigned int nominator, unsigned int denominator, Accelerator* accelerator) {
    this->nominator = nominator;
    this->denominator = denominator;
    this->accelerator = accelerator;
}

unsigned int AcceleratorContext::divCpu(unsigned int value) {
    return value - divAcc(value);
}

unsigned int AcceleratorContext::divAcc(unsigned int value) {
    return (nominator * value) / denominator;
}

MatmulCommand::MatmulCommand(const unsigned int n, const unsigned int d, const FloatType weightsFloatType, const FloatType inputFloatType, AcceleratorContext* acc) {
    this->n = n;
    this->d = d;
    this->inputFloatType = inputFloatType;
    this->weightsFloatType = weightsFloatType;
    this->acc = acc;
    this->accD = acc->divAcc(d);
    this->accSize = getBatchBytes(weightsFloatType, n, this->accD);
    this->cpuD = acc->divCpu(d);
    this->cpuSize = getBatchBytes(weightsFloatType, n, this->cpuD);

    // printf("matmulCommand cpuD=%d, cpuSize=%zu, accD=%d, accSize=%zu\n", cpuD, cpuSize, accD, accSize);
    if (this->cpuD != 0)
        this->cpuWeights = newBuffer(this->cpuSize);
    if (this->accD != 0) {
        this->accMatmulIndex = acc->accelerator->allocateMatmul(weightsFloatType, inputFloatType, n, this->accD);
    }
};

MatmulCommand::~MatmulCommand() {
    if (this->cpuD != 0)
        freeBuffer(cpuWeights);
}

size_t MatmulCommand::loadWeights(const void* source) {
    if (cpuD != 0) {
        memcpy(cpuWeights, source, cpuSize);
    }
    if (accD != 0) {
        acc->accelerator->loadMatmulWeights(accMatmulIndex, &((char*)source)[cpuSize]);
    }
    return cpuSize + accSize;
}

void MatmulCommand::forward(const void* input, float* output, const unsigned int nThreads, const unsigned int threadIndex) {
    if (accD != 0 && threadIndex == 0) {
        acc->accelerator->beginForwardMatmul(accMatmulIndex, input);
    }
    if (cpuD != 0) {
        matmul(weightsFloatType, inputFloatType, output, input, cpuWeights, n, cpuD, nThreads, threadIndex);
    }
    if (accD != 0 && threadIndex == 0) {
        float* gpuOutput = &output[cpuD];
        acc->accelerator->endForwardMatmul(accMatmulIndex, gpuOutput);
        // DEBUG_FLOATS("gpu", gpuOutput, 8);
    }
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

void LlamaRopeCommand::forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex) {
    const unsigned int dim0Half = (isQ ? slice->qDim0 : slice->kvDim0) / 2;
    const unsigned int shift = isQ ? slice->qShift : 0;
    SPLIT_RANGE_TO_THREADS(s, e, 0, dim0Half, nThreads, threadIndex);
    const unsigned int iStart = s * 2;
    const unsigned int iEnd = e * 2;

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

void FalconRopeCommand::forward(bool isQ, float* qOrK, pos_t pos, unsigned int nThreads, unsigned int threadIndex) {
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