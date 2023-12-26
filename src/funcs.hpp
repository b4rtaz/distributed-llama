#ifndef FUNCS_HPP
#define FUNCS_HPP

#include "quants.hpp"

void softmax(float* x, const int size);
void rmsnorm(float* o, const float* x, const float* weight, const int size);
void matmul(FloatType type, float* output, float* input, void* weights, int n, int d, unsigned int nThreads, unsigned int threadIndex);
float dotProduct(const float* a, const float* b, const int size);

#endif
