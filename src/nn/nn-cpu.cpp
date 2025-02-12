#include "nn-cpu.hpp"
#include "nn-cpu-ops.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define DEBUG_CPU_OP_QUANTS false

#define BUFFER_ALIGNMENT 16

static NnByte *allocAlignedBuffer(size_t size) {
    NnByte *buffer;
#ifdef _WIN32
    buffer = (NnByte *)_aligned_malloc(size, BUFFER_ALIGNMENT);
    if (buffer == NULL)
        throw std::runtime_error("_aligned_malloc failed");
#else
    if (posix_memalign((void **)&buffer, BUFFER_ALIGNMENT, size) != 0)
        throw std::runtime_error("posix_memalign failed");
    if (mlock(buffer, size) != 0)
        fprintf(stderr, "🚧 Cannot allocate %zu bytes directly in RAM\n", size);
#endif
    return buffer;
}

static void releaseAlignedBuffer(NnByte *buffer) {
#ifdef _WIN32
    _aligned_free(buffer);
#else
    free(buffer);
#endif
}

NnCpuDevice::NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->netExecution = netExecution;

    printCpuInstructionSet();

    nBuffers = nodeConfig->nBuffers;
    buffers = new NnByte *[nBuffers];
    for (NnSize bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++) {
        NnBufferConfig *config = &nodeConfig->buffers[bufferIndex];
        NnByte *buffer = allocAlignedBuffer(config->size.nBytes);
        buffers[bufferIndex] = buffer;
    }

    bufferFlags = new NnByte[nBuffers];
    std::memset(bufferFlags, 0, nBuffers * sizeof(NnByte));
}

NnCpuDevice::~NnCpuDevice() {
    for (NnSize bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++)
        releaseAlignedBuffer(buffers[bufferIndex]);
    delete[] buffers;
    delete[] bufferFlags;
}

NnSize NnCpuDevice::maxNThreads() {
    return std::thread::hardware_concurrency();
}

NnDeviceSegment *NnCpuDevice::createSegment(NnSize segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    assert(segmentConfig->nOps > 0);

    std::vector<NnOpQuantType> opQuants(segmentConfig->nOps);
    std::vector<NnCpuOpForward> opForwardLocal(segmentConfig->nOps);
    std::vector<NnSize2D> inputSizes(segmentConfig->nOps);
    std::vector<NnSize2D> outputSizes(segmentConfig->nOps);

    std::unique_ptr<NnByte *[]> inputsPtr(new NnByte *[segmentConfig->nOps * netConfig->nBatches]);
    std::unique_ptr<NnByte *[]> outputsPtr(new NnByte *[segmentConfig->nOps * netConfig->nBatches]);
    NnByte **inputs = inputsPtr.get();
    NnByte **outputs = outputsPtr.get();

    for (NnSize opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize2D inputSize;
        NnSize2D outputSize;
        resolvePointer(&inputs[opIndex * netConfig->nBatches], &inputSize, &opConfig->input);
        resolvePointer(&outputs[opIndex * netConfig->nBatches], &outputSize, &opConfig->output);
        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType);
        #if DEBUG_CPU_OP_QUANTS
            printf("%20s %2d: %s\n", opConfig->name, opConfig->index, opQuantTypeToString(opQuant));
        #endif
        NnCpuOpForward forward = getCpuOpForward(opConfig->code, opQuant);
        if (forward == nullptr) {
            throw std::invalid_argument(
                std::string("Unsupported CPU op code: ") + opCodeToString(opConfig->code) + 
                ", quant: " + opQuantTypeToString(opQuant) +
                ", op name: " + opConfig->name);
        }
        inputSizes[opIndex] = inputSize;
        outputSizes[opIndex] = outputSize;
        opQuants[opIndex] = opQuant;
        opForwardLocal[opIndex] = forward;
    }

    inputsPtr.release();
    outputsPtr.release();

    NnCpuOpForward *opForward = new NnCpuOpForward[segmentConfig->nOps];
    NnCpuOpContext *opContexts = new NnCpuOpContext[segmentConfig->nOps];

    for (NnSize opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnCpuOpContext *opContext = &opContexts[opIndex];
        NnCpuOpForwardInit opInit = getCpuOpForwardInit(opConfig->code, opQuants[opIndex]);
        opContext->name = opConfig->name;
        opContext->opConfig = opConfig->config;
        opContext->weightSize = opConfig->weightSize;
        opContext->nBatches = netConfig->nBatches;
        opContext->pipes = netExecution->pipes;
        opContext->pipeConfigs = netConfig->pipes;
        opContext->buffers = buffers;
        opContext->bufferConfigs = nodeConfig->buffers;
        opContext->bufferFlags = bufferFlags;

        opContext->input = &inputs[opIndex * netConfig->nBatches];
        opContext->inputSize = inputSizes[opIndex];
        opContext->hasInputContinuousMemory = hasPointerContinuousMemory(&opConfig->input);

        opContext->output = &outputs[opIndex * netConfig->nBatches];
        opContext->outputSize = outputSizes[opIndex];
        opContext->hasOutputContinuousMemory = hasPointerContinuousMemory(&opConfig->output);

        if (opContext->weightSize.nBytes > 0)
            opContext->weight = allocAlignedBuffer(opContext->weightSize.nBytes);
        else
            opContext->weight = nullptr;

        if (opInit != nullptr)
            opInit(opContext);
        opForward[opIndex] = opForwardLocal[opIndex];
    }
    return new NnCpuDeviceSegment(opForward, opContexts, segmentConfig->nOps);
}

NnCpuDeviceSegment::~NnCpuDeviceSegment() {
    for (NnSize opIndex = 0; opIndex < nOps; opIndex++) {
        NnCpuOpContext *context = &opContexts[opIndex];
        if (opIndex == 0) {
            delete[] context->input;
            delete[] context->output;
        }
        if (context->weightSize.nBytes > 0)
            releaseAlignedBuffer(context->weight);
    }
    delete[] opForward;
    delete[] opContexts;
}

void NnCpuDevice::resolvePointer(NnByte **pntr, NnSize2D *pntrSize, NnPointerConfig *pointerConfig) {
    NnByte *source;
    NnSize2D *sourceSize;
    if (pointerConfig->pointerType == PNTR_BUFFER) {
        source = buffers[pointerConfig->pointerIndex];
        sourceSize = &nodeConfig->buffers[pointerConfig->pointerIndex].size;
    } else if (pointerConfig->pointerType == PNTR_PIPE) {
        source = netExecution->pipes[pointerConfig->pointerIndex];
        sourceSize = &netConfig->pipes[pointerConfig->pointerIndex].size;
    } else {
        throw std::invalid_argument("Unsupported pointer type");
    }

    if (pointerConfig->batchType == PNTR_BATCH_DEFAULT) {
        ASSERT_EQ(sourceSize->y, netConfig->nBatches);

        NnSize batchBytes = getBytes(sourceSize->floatType, sourceSize->x);
        for (NnSize batchIndex = 0; batchIndex < netConfig->nBatches; batchIndex++)
            pntr[batchIndex] = &source[batchIndex * batchBytes];
        *pntrSize = *sourceSize;

        if (pointerConfig->sliceType == SLICE_NONE)
            return;
        if (pointerConfig->sliceType == SLICE_NODE_PART) {
            assert(sourceSize->x % netConfig->nNodes == 0);
            NnSize xSlice = sourceSize->x / netConfig->nNodes;
            NnSize xSliceBytes = getBytes(sourceSize->floatType, xSlice);
            for (NnSize batchIndex = 0; batchIndex < netConfig->nBatches; batchIndex++)
                pntr[batchIndex] = &pntr[batchIndex][xSliceBytes * nodeConfig->nodeIndex];
            *pntrSize = size2D(sourceSize->floatType, sourceSize->y, xSlice);
            return;
        }
    }
    if (pointerConfig->batchType == PNTR_BATCH_PIPE) {
        if (pointerConfig->sliceType == SLICE_NONE) {
            dynamicPointers.push_back({ source, sourceSize, pntr, pointerConfig });
            *pntrSize = size2D(sourceSize->floatType, netConfig->nBatches, sourceSize->x);
            return;
        }
    }
    throw std::invalid_argument("Unsupported pointer config");
}

void NnCpuDevice::syncPointers() {
    NnSize nDynamicPointers = dynamicPointers.size();
    for (NnSize dynamicPointerIndex = 0; dynamicPointerIndex < nDynamicPointers; dynamicPointerIndex++) {
        NnCpuDynamicPointer *dp = &dynamicPointers[dynamicPointerIndex];
        assert(dp->pointerConfig->batchType == PNTR_BATCH_PIPE);
        float *pipe = (float *)netExecution->pipes[dp->pointerConfig->batchArg0];

        for (NnSize batchIndex = 0; batchIndex < netExecution->batchSize; batchIndex++) {
            NnSize index = (NnSize)pipe[batchIndex];
            assert(index < dp->sourceSize->y);
            NnSize nBytes = dp->sourceSize->nBytes / dp->sourceSize->y;
            dp->pntr[batchIndex] = &dp->source[index * nBytes];
        }
    }
}

void NnCpuDeviceSegment::loadWeight(NnSize opIndex, NnSize nBytes, NnByte *weight) {
    assert(opIndex >= 0);
    assert(opIndex < nOps);
    NnCpuOpContext *context = &opContexts[opIndex];
    ASSERT_EQ(context->weightSize.nBytes, nBytes);
    std::memcpy(context->weight, weight, nBytes);
}

void NnCpuDeviceSegment::forward(NnSize opIndex, NnSize nThreads, NnSize threadIndex, NnSize batchSize) {
    NnCpuOpContext *context = &opContexts[opIndex];
    // printf("forward: %d %s (%d/%d)\n", opIndex, context->name, threadIndex + 1, nThreads); fflush(stdout);
    opForward[opIndex](nThreads, threadIndex, batchSize, context);
}
