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
    NnByte **input;
    NnSize2D inputSize;
    NnByte **output;
    NnSize2D outputSize;
    NnByte *weight;
    NnSize2D weightSize;
    void *opConfig;
} NnCpuOpContext;

typedef void (*NnCpuOpForwardInit)(NnCpuOpContext *context);
typedef void (*NnCpuOpForward)(NnSize nThreads, NnSize threadIndex, NnSize batchSize, NnCpuOpContext *context);

NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType);
NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType);

#endif