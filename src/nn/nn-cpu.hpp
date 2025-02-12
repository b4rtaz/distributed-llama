#ifndef NN_CPU_H
#define NN_CPU_H

#include <vector>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

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
    NnSize nBuffers;
    NnByte *bufferFlags;
    std::vector<NnCpuDynamicPointer> dynamicPointers;
public:
    NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnCpuDevice();
    NnSize maxNThreads() override;
    NnDeviceSegment *createSegment(NnSize segmentIndex) override;
    void syncPointers() override;
    void resolvePointer(NnByte **pntr, NnSize2D *pntrSize, NnPointerConfig *pointerConfig);
};

class NnCpuDeviceSegment : public NnDeviceSegment {
public:
    NnSize nOps;
    NnCpuOpForward *opForward;
    NnCpuOpContext *opContexts;
    NnCpuDeviceSegment(NnCpuOpForward *opForward, NnCpuOpContext *opContexts, NnSize nOps)
        : opForward(opForward), opContexts(opContexts), nOps(nOps) {}
    ~NnCpuDeviceSegment() override;
    void loadWeight(NnSize opIndex, NnSize nBytes, NnByte *weight) override;
    void forward(NnSize opIndex, NnSize nThreads, NnSize threadIndex, NnSize batchSize) override;
};

#endif