#ifndef NN_EXECUTOR_H
#define NN_EXECUTOR_H

#include "nn-core.hpp"
#include <atomic>
#include <vector>
#include "pthread.h"

class NnDeviceSegment {
public:
    virtual ~NnDeviceSegment() {};
    virtual void loadWeight(NnSize opIndex, NnSize nBytes, NnByte *weight) = 0;
    virtual void forward(NnSize opIndex, NnSize nThreads, NnSize threadIndex, NnSize batchSize) = 0;
};

class NnDevice {
public:
    virtual NnSize maxNThreads() = 0;
    virtual NnDeviceSegment *createSegment(NnSize segmentIndex) = 0;
    virtual void syncPointers() = 0;
};

class NnNodeSynchronizer {
public:
    virtual ~NnNodeSynchronizer() {};
    virtual void sync(NnSize segmentIndex, NnSize nThreads, NnSize threadIndex) = 0;
};

class NnFakeNodeSynchronizer : public NnNodeSynchronizer {
public:
    ~NnFakeNodeSynchronizer() override {};
    void sync(NnSize segmentIndex, NnSize nThreads, NnSize threadIndex) override;
};

class NnNetExecution {
public:
    NnSize nThreads;
    NnByte **pipes;
    NnSize batchSize;
private:
    NnSize nBatches;
    NnSize nPipes;
public:
    NnNetExecution(NnSize nThreads, NnNetConfig *netConfig);
    ~NnNetExecution();
    void setBatchSize(NnSize batchSize);
};

enum NnExecutorStepType {
    STEP_EXECUTE_OP,
    STEP_SYNC_NODES,
    STEP_SYNC_POINTERS
};

typedef struct {
    NnExecutorStepType type;
    NnDeviceSegment *segment;
    NnSize arg0;
    NnOpConfig *opConfig;
} NnExecutorStep;

typedef struct {
    NnSize nThreads;
    NnSize nSteps;
    NnExecutorStep *steps;
    NnNodeSynchronizer *synchronizer;
    NnDevice *device;
    std::atomic_uint currentStepIndex;
    std::atomic_uint doneThreadCount;
    NnSize batchSize;
} NnExecutorContext;

typedef struct {
    NnSize threadIndex;
    NnExecutorContext *context;
    PthreadHandler handler;
} NnExecutorThread;

class NnExecutor {
public:
    NnNetExecution *netExecution;
    NnNodeConfig *nodeConfig;
    std::vector<std::unique_ptr<NnDeviceSegment>> segments;
    std::vector<NnExecutorStep> steps;
    NnExecutorThread *threads;
    NnExecutorContext context;
    NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnDevice *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer);
    ~NnExecutor();
    void loadWeight(const char *name, NnSize index, NnSize nBytes, NnByte *weight);
    void forward();
};

#endif