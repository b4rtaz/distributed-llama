#ifndef FUNCS_HPP
#define FUNCS_HPP

#include "quants.hpp"

void softmax(float* x, const int size);
float rms(const float* x, const int size);
void rmsnorm(float* o, const float* x, const float ms, const float* weight, const int size, unsigned int nThreads, unsigned int threadIndex);
void matmul(FloatType weightsFloatType, FloatType inputFloatType, float* output, void* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex);
float dotProduct(const float* a, const float* b, const int size);

#endif
