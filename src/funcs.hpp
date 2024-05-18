#ifndef FUNCS_HPP
#define FUNCS_HPP

#include "quants.hpp"

#define SPLIT_RANGE_TO_THREADS(varStart, varEnd, rangeStart, rangeEnd, nThreads, threadIndex) \
    const unsigned int rangeLen = (rangeEnd - rangeStart); \
    const unsigned int rangeSlice = rangeLen / nThreads; \
    const unsigned int rangeRest = rangeLen % nThreads; \
    const unsigned int varStart = threadIndex * rangeSlice + (threadIndex < rangeRest ? threadIndex : rangeRest); \
    const unsigned int varEnd = varStart + rangeSlice + (threadIndex < rangeRest ? 1 : 0);

void softmax(float* x, const unsigned int size);
float rms(const float* x, const unsigned int size);
void rmsnorm(float* o, const float* x, const float ms, const float* weight, const unsigned int size, const unsigned int nThreads, const unsigned int threadIndex);
void matmul(const FloatType weightsFloatType, const FloatType inputFloatType, float* output, const void* input, const void* weights, const unsigned int n, const unsigned int d, const unsigned int nThreads, const unsigned int threadIndex);
float dotProduct(const float* a, const float* b, const unsigned int size);
void gelu(float* t, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void silu(float* t, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void mul(float* output, const float* input, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void mulScalar(float* output, const float c, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void add(float* output, const float* input, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);

#endif
