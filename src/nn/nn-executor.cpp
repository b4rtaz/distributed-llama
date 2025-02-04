#include <cassert>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include "nn-executor.hpp"

#define DEBUG_BENCHMARK false

void NnFakeNodeSynchronizer::sync(NnSize segmentIndex, NnSize nThreads, NnSize threadIndex) {
    // Nothing
}

NnNetExecution::NnNetExecution(NnSize nThreads, NnNetConfig *netConfig) {
    this->nThreads = nThreads;
    this->nBatches = netConfig->nBatches;
    this->nPipes = netConfig->nPipes;
    this->batchSize = 0; // This value must be overwritten before calling forward

    pipes = new NnByte *[netConfig->nPipes];
    for (NnSize pipeIndex = 0; pipeIndex < netConfig->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &netConfig->pipes[pipeIndex];
        NnByte *pipe = new NnByte[pipeConfig->size.nBytes];
        std::memset(pipe, 0, pipeConfig->size.nBytes);
        pipes[pipeIndex] = pipe;
    }
}

NnNetExecution::~NnNetExecution() {
    for (NnSize pipeIndex = 0; pipeIndex < nPipes; pipeIndex++)
        delete[] pipes[pipeIndex];
    delete[] pipes;
}

void NnNetExecution::setBatchSize(NnSize batchSize) {
    assert(batchSize > 0 && batchSize <= nBatches);
    this->batchSize = batchSize;
}

NnExecutor::NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnDevice *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer) {
    if (netExecution->nThreads > device->maxNThreads())
        throw std::invalid_argument("This device does not support that many threads");
    this->netExecution = netExecution;
    this->nodeConfig = nodeConfig;

    bool useSynchronizer = netConfig->nNodes > 1;

    context.nThreads = netExecution->nThreads;
    context.synchronizer = synchronizer;
    context.device = device;
    context.nSteps = 0;

    std::unique_ptr<NnDeviceSegment *> deviceSegmentsPtr(new NnDeviceSegment *[nodeConfig->nSegments]);
    NnDeviceSegment **deviceSegments = deviceSegmentsPtr.get();

    for (NnSize segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        if (segmentConfig->nOps > 0) {
            deviceSegments[segmentIndex] = device->createSegment(segmentIndex);
            context.nSteps += segmentConfig->nOps;
        }
        if (useSynchronizer && segmentConfig->nSyncs > 0)
            context.nSteps++;
        if (segmentConfig->syncPointers)
            context.nSteps++;
    }
    assert(context.nSteps > 0);

    segments = deviceSegmentsPtr.release();
    context.steps = new NnExecutorStep[context.nSteps];

    NnSize currentSegmentIndex = 0;
    NnSize currentOpIndex = 0;
    for (NnSize stepIndex = 0; stepIndex < context.nSteps;) {
        NnDeviceSegment *segment = segments[currentSegmentIndex];
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[currentSegmentIndex];
        if (currentOpIndex >= segmentConfig->nOps) {
            if (useSynchronizer && segmentConfig->nSyncs > 0) {
                context.steps[stepIndex] = { STEP_SYNC_NODES, nullptr, currentSegmentIndex, nullptr };
                stepIndex++;
            }
            if (segmentConfig->syncPointers) {
                context.steps[stepIndex] = { STEP_SYNC_POINTERS, nullptr, 0, nullptr };
                stepIndex++;
            }
            currentSegmentIndex++;
            currentOpIndex = 0;
            segment = segments[currentSegmentIndex];
        }
        NnOpConfig *opConfig = &segmentConfig->ops[currentOpIndex];
        context.steps[stepIndex] = { STEP_EXECUTE_OP, segment, currentOpIndex, opConfig };
        currentOpIndex++;
        stepIndex++;
    }

    threads = new NnExecutorThread[netExecution->nThreads];
    for (NnSize threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        NnExecutorThread *thread = &threads[threadIndex];
        thread->threadIndex = threadIndex;
        thread->context = &context;
    }
}

NnExecutor::~NnExecutor() {
    for (NnSize segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++)
        delete segments[segmentIndex];
    delete[] context.steps;
    delete[] segments;
    delete[] threads;
}

void NnExecutor::loadWeight(const char *name, NnSize index, NnSize nBytes, NnByte *weight) {
    for (NnSize segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        for (NnSize opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (std::strcmp(opConfig->name, name) == 0 && opConfig->index == index) {
                segments[segmentIndex]->loadWeight(opIndex, nBytes, weight);
                return;
            }
        }
    }
    throw std::invalid_argument("Cannot locate op by name: " + std::string(name));
}

inline void executeStep(NnExecutorStep *step, NnSize nThreads, NnExecutorThread *thread, NnExecutorContext *context) {
    #if DEBUG_BENCHMARK
    assert(nThreads == 1);
    auto startTime = std::chrono::high_resolution_clock::now();
    #endif

    if (step->type == STEP_EXECUTE_OP) {
        step->segment->forward(step->arg0, nThreads, thread->threadIndex, context->batchSize);
    } else if (step->type == STEP_SYNC_NODES) {
        context->synchronizer->sync(step->arg0, nThreads, thread->threadIndex);
    } else if (step->type == STEP_SYNC_POINTERS) {
        if (thread->threadIndex == 0)
            context->device->syncPointers();
    } else {
        throw std::invalid_argument("Unsupported step type");
    }

    #if DEBUG_BENCHMARK
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    if (step->type == STEP_EXECUTE_OP)
        printf("🕒 [OP %16s %2d] %6lld μs\n", opCodeToString(step->opConfig->code), step->opConfig->index, duration);
    else if (step->type == STEP_SYNC_NODES)
        printf("🕒 [SYNC %17d] %6lld μs\n", step->arg0, duration);
    #endif
}

static inline void *executorThreadHandler(void *arg) {
    NnExecutorThread *thread = (NnExecutorThread *)arg;
    NnExecutorContext *context = thread->context;
    NnSize nThreads = context->nThreads;
    NnSize doneCount = nThreads - 1;

    while (true) {
        const unsigned int currentStepIndex = context->currentStepIndex.load();
        if (currentStepIndex == context->nSteps)
            break;

        NnExecutorStep *step = &context->steps[currentStepIndex];
        executeStep(step, nThreads, thread, context);

        NnSize currentCount = context->doneThreadCount.fetch_add(1);
        if (currentCount == doneCount) {
            context->doneThreadCount.store(0);
            context->currentStepIndex.fetch_add(1);
        } else {
            while (context->currentStepIndex.load() == currentStepIndex);
        }
    }
    return nullptr;
}

void NnExecutor::forward() {
    assert(netExecution->batchSize > 0);

    NnSize nThreads = netExecution->nThreads;
    context.currentStepIndex.exchange(0);
    context.doneThreadCount.exchange(0);
    context.batchSize = netExecution->batchSize;

    NnSize threadIndex;
    for (threadIndex = 1; threadIndex < nThreads; threadIndex++) {
        int result = pthread_create(&threads[threadIndex].handler, NULL, (PthreadFunc)executorThreadHandler, (void *)&threads[threadIndex]);
        if (result != 0)
            throw std::runtime_error("Failed to create thread");
    }
    executorThreadHandler((void *)&threads[0]);
    for (threadIndex = 1; threadIndex < nThreads; threadIndex++)
        pthread_join(threads[threadIndex].handler, NULL);
}
