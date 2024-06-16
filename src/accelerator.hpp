#ifndef ACCELERATOR_HPP
#define ACCELERATOR_HPP

#include "quants.hpp"

class Accelerator {
public:
    virtual ~Accelerator() {}
    virtual unsigned int allocateMatmul(const FloatType weightsFloatType, const FloatType inputFloatType, const unsigned int n, const unsigned int d) = 0;
    virtual void loadMatmulWeights(const unsigned int matmulIndex, const void* weights) = 0;
    virtual void beginForwardMatmul(const unsigned int matmulIndex, const void* input) = 0;
    virtual void endForwardMatmul(const unsigned int matmulIndex, float* output) = 0;
};

#endif
