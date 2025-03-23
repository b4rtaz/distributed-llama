#ifndef NN_EXECUTOR_H
#define NN_EXECUTOR_H

#include "nn-core.hpp"
#include <atomic>
#include <vector>
#include "pthread.h"

class NnDeviceSegment {
public:
    virtual ~NnDeviceSegment() {};
    virtual void loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) = 0;
    virtual void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) = 0;
};

class NnDevice {
public:
    virtual NnUint maxNThreads() = 0;
    virtual ~NnDevice() {}
    virtual NnDeviceSegment *createSegment(NnUint segmentIndex) = 0;
};

class NnNodeSynchronizer {
public:
    virtual ~NnNodeSynchronizer() {};
    virtual void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) = 0;
};

class NnFakeNodeSynchronizer : public NnNodeSynchronizer {
public:
    ~NnFakeNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

class NnNetExecution {
public:
    NnUint nThreads;
    NnUint nPipes;
    NnByte **pipes;
    NnUint batchSize;
    NnUint nBatches;
    NnNetExecution(NnUint nThreads, NnNetConfig *netConfig);
    ~NnNetExecution();
    void setBatchSize(NnUint batchSize);
};

enum NnExecutorStepType {
    STEP_EXECUTE_OP,
    STEP_SYNC_NODES,
};

#define N_STEP_TYPES STEP_SYNC_NODES + 1

typedef struct {
    NnExecutorStepType type;
    NnDeviceSegment *segment;
    NnUint arg0;
    NnOpConfig *opConfig;
} NnExecutorStep;

typedef struct {
    NnUint nThreads;
    NnUint nSteps;
    NnExecutorStep *steps;
    NnNodeSynchronizer *synchronizer;
    NnDevice *device;
    std::atomic_uint currentStepIndex;
    std::atomic_uint doneThreadCount;
    NnUint batchSize;
    Timer *timer;
    NnUint totalTime[N_STEP_TYPES];
} NnExecutorContext;

typedef struct {
    NnUint threadIndex;
    NnExecutorContext *context;
    PthreadHandler handler;
} NnExecutorThread;

class NnExecutor {
private:
    NnNetExecution *netExecution;
    NnNodeConfig *nodeConfig;
    std::vector<std::unique_ptr<NnDeviceSegment>> segments;
    std::vector<NnExecutorStep> steps;
    NnExecutorThread *threads;
    NnExecutorContext context;
public:
    NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnDevice *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark);
    ~NnExecutor();
    void loadWeight(const char *name, NnUint index, NnSize nBytes, NnByte *weight);
    void forward();
    NnUint getTotalTime(NnExecutorStepType type);
};

#endif