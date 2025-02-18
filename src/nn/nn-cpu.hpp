#ifndef NN_CPU_H
#define NN_CPU_H

#include <vector>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

#define DEBUG_USE_MMAP_FOR_WEIGHTS false

typedef struct {
    NnByte *source;
    NnSize2D *sourceSize;
    NnByte **pntr;
    NnPointerConfig *pointerConfig;
} NnCpuDynamicPointer;

class NnCpuDevice : public NnDevice {
public:
    NnByte **buffers;
private:
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
    NnUint nBuffers;
    NnByte *bufferFlags;
    std::vector<NnCpuDynamicPointer> dynamicPointers;
public:
    NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnCpuDevice();
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    void syncPointers() override;
    void resolvePointer(NnByte **pntr, NnSize2D *pntrSize, NnPointerConfig *pointerConfig);
};

class NnCpuDeviceSegment : public NnDeviceSegment {
public:
    NnUint nOps;
    NnCpuOpForward *opForward;
    NnCpuOpContext *opContexts;
    NnCpuDeviceSegment(NnCpuOpForward *opForward, NnCpuOpContext *opContexts, NnUint nOps)
        : opForward(opForward), opContexts(opContexts), nOps(nOps) {}
    ~NnCpuDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif