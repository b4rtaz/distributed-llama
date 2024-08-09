#ifndef FUNCS_HPP
#define FUNCS_HPP

#include "quants.hpp"

// 输出数组x的softmax结果
void softmax(float* x, const unsigned int size);
// 输出数组x的均方根结果
float rms(const float* x, const unsigned int size);

void rmsnorm(float* o, const float* x, const float ms, const float* weight, const unsigned int size, const unsigned int nThreads, const unsigned int threadIndex);
void matmul(const FloatType weightsFloatType, const FloatType inputFloatType, float* output, const void* input, const void* weights, const unsigned int n, const unsigned int d, const unsigned int nThreads, const unsigned int threadIndex, int memoryBudget, int FLAG);
float dotProduct(const float* a, const float* b, const unsigned int size);
void gelu(float* t, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void silu(float* t, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void mul(float* output, const float* input, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void mulScalar(float* output, const float c, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);
void add(float* output, const float* input, const unsigned int n, const unsigned int nThreads, const unsigned int threadIndex);

#endif
