#include <cassert>
#include <cstring>
#include <stdexcept>
#include "nn-executor.hpp"

#define DEBUG_EXECUTOR_BENCHMARK false

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

NnExecutor::NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnDevice *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer)
    : segments(nodeConfig->nSegments), steps()
{
    NnSize maxNThreads = device->maxNThreads();
    if (netExecution->nThreads > maxNThreads)
        throw std::invalid_argument("This CPU supports max " + std::to_string(maxNThreads) + " threads");
    this->netExecution = netExecution;
    this->nodeConfig = nodeConfig;

    bool useSynchronizer = netConfig->nNodes > 1;
    for (NnSize segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        if (segmentConfig->nOps > 0) {
            NnDeviceSegment *segment = device->createSegment(segmentIndex);
            segments[segmentIndex] = std::unique_ptr<NnDeviceSegment>(segment);
    
            for (NnSize opIndex = 0; opIndex < segmentConfig->nOps; opIndex++)
                steps.push_back(NnExecutorStep{ STEP_EXECUTE_OP, segment, opIndex, &segmentConfig->ops[opIndex] });
        }
        if (useSynchronizer && segmentConfig->nSyncs > 0)
            steps.push_back(NnExecutorStep{ STEP_SYNC_NODES, nullptr, segmentIndex, nullptr });
        if (segmentConfig->syncPointers)
            steps.push_back(NnExecutorStep{ STEP_SYNC_POINTERS, nullptr, 0, nullptr });
    }

    steps.shrink_to_fit();

    context.nThreads = netExecution->nThreads;
    context.synchronizer = synchronizer;
    context.device = device;
    context.nSteps = (NnSize)steps.size();
    context.steps = steps.data();

    threads = new NnExecutorThread[netExecution->nThreads];
    for (NnSize threadIndex = 0; threadIndex < netExecution->nThreads; threadIndex++) {
        NnExecutorThread *thread = &threads[threadIndex];
        thread->threadIndex = threadIndex;
        thread->context = &context;
    }
}

NnExecutor::~NnExecutor() {
    delete[] threads;
}

void NnExecutor::loadWeight(const char *name, NnSize index, NnSize nBytes, NnByte *weight) {
    for (NnSize segmentIndex = 0; segmentIndex < nodeConfig->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
        for (NnSize opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->index == index && std::strcmp(opConfig->name, name) == 0) {
                NnDeviceSegment *segment = segments[segmentIndex].get();
                assert(segment != nullptr);
                segment->loadWeight(opIndex, nBytes, weight);
                return;
            }
        }
    }
    throw std::invalid_argument("Cannot locate op by name: " + std::string(name));
}

inline void executeStep(NnExecutorStep *step, NnSize nThreads, NnExecutorThread *thread, NnExecutorContext *context) {
#if DEBUG_EXECUTOR_BENCHMARK
    assert(nThreads == 1);
    Timer startTime;
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

#if DEBUG_EXECUTOR_BENCHMARK
    NnSize duration = startTime.elapsedMicroseconds();
    if (step->type == STEP_EXECUTE_OP)
        printf("🕒 [OP %16s %2d] %u μs\n", opCodeToString(step->opConfig->code), step->opConfig->index, duration);
    else if (step->type == STEP_SYNC_NODES)
        printf("🕒 [SYNC %17d] %u μs\n", step->arg0, duration);
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
