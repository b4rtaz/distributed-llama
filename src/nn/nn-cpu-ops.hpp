#ifndef NN_CPU_OPS_H
#define NN_CPU_OPS_H

#include "nn-core.hpp"

#define ASSERT_EQ(a, b) \
    if (a != b) { \
        printf("Assertion failed: %d != %d (%s:%d)\n", a, b, __FILE__, __LINE__); \
        exit(-1); \
    }

typedef struct {
    const char *name;
    NnByte nBatches;
    NnByte *bufferFlags;
    NnByte **buffers;
    NnBufferConfig *bufferConfigs;
    NnByte **pipes;
    NnPipeConfig *pipeConfigs;
    void *opConfig;

    NnByte **input;
    NnSize2D inputSize;
    bool hasInputContinuousMemory;

    NnByte **output;
    NnSize2D outputSize;
    bool hasOutputContinuousMemory;

    NnByte *weight;
    NnSize2D weightSize;
} NnCpuOpContext;

typedef void (*NnCpuOpForwardInit)(NnCpuOpContext *context);
typedef void (*NnCpuOpForward)(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context);

void printCpuInstructionSet();
NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType);
NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType);

void softmax_F32(float *x, const NnUint size);

#endif