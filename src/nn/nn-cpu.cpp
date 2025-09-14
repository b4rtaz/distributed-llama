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

#define BUFFER_ALIGNMENT 64

static NnByte *allocAlignedBuffer(NnSize size) {
    NnByte *buffer;
#ifdef _WIN32
    buffer = (NnByte *)_aligned_malloc(size, BUFFER_ALIGNMENT);
    if (buffer == NULL)
        throw std::runtime_error("_aligned_malloc failed");
#else
    if (posix_memalign((void **)&buffer, BUFFER_ALIGNMENT, size) != 0)
        throw std::runtime_error("posix_memalign failed");
    mlock(buffer, size);
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
    for (NnUint bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++) {
        NnBufferConfig *config = &nodeConfig->buffers[bufferIndex];
        NnByte *buffer = allocAlignedBuffer(config->size.nBytes);
        buffers[bufferIndex] = buffer;
    }

    bufferFlags = new NnByte[nBuffers];
    std::memset(bufferFlags, 0, nBuffers * sizeof(NnByte));
}

NnCpuDevice::~NnCpuDevice() {
    for (NnUint bufferIndex = 0; bufferIndex < nBuffers; bufferIndex++)
        releaseAlignedBuffer(buffers[bufferIndex]);
    delete[] buffers;
    delete[] bufferFlags;
}

NnUint NnCpuDevice::maxNThreads() {
    return std::thread::hardware_concurrency();
}

NnDeviceSegment *NnCpuDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    assert(segmentConfig->nOps > 0);

    std::vector<NnOpQuantType> opQuants(segmentConfig->nOps);
    std::vector<NnCpuOpForward> opForwardLocal(segmentConfig->nOps);
    std::vector<NnSize3D> inputSizes(segmentConfig->nOps);
    std::vector<NnSize3D> outputSizes(segmentConfig->nOps);

    std::vector<std::vector<NnByte *>> inputsPtr(segmentConfig->nOps);
    std::vector<std::vector<NnByte *>> outputsPtr(segmentConfig->nOps);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize3D inputSize;
        NnSize3D outputSize;
        inputsPtr[opIndex] = resolvePointer(&inputSize, &opConfig->input);
        outputsPtr[opIndex] = resolvePointer(&outputSize, &opConfig->output);
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

    NnCpuOpForward *opForward = new NnCpuOpForward[segmentConfig->nOps];
    NnCpuOpContext *opContexts = new NnCpuOpContext[segmentConfig->nOps];

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
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

        opContext->input = new NnByte *[inputsPtr[opIndex].size()];
        opContext->inputSize = inputSizes[opIndex];
        opContext->hasInputContinuousMemory = hasPointerContinuousMemory(&opConfig->input);
        std::memcpy(opContext->input, inputsPtr[opIndex].data(), inputsPtr[opIndex].size() * sizeof(NnByte *));

        opContext->output = new NnByte *[outputsPtr[opIndex].size()];
        opContext->outputSize = outputSizes[opIndex];
        opContext->hasOutputContinuousMemory = hasPointerContinuousMemory(&opConfig->output);
        std::memcpy(opContext->output, outputsPtr[opIndex].data(), outputsPtr[opIndex].size() * sizeof(NnByte *));

#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
        if (opContext->weightSize.nBytes > 0)
            opContext->weight = allocAlignedBuffer(opContext->weightSize.nBytes);
        else
            opContext->weight = nullptr;
#endif

        if (opInit != nullptr)
            opInit(opContext);
        opForward[opIndex] = opForwardLocal[opIndex];
    }
    return new NnCpuDeviceSegment(opForward, opContexts, segmentConfig->nOps);
}

NnCpuDeviceSegment::~NnCpuDeviceSegment() {
    for (NnUint opIndex = 0; opIndex < nOps; opIndex++) {
        NnCpuOpContext *context = &opContexts[opIndex];
        delete[] context->input;
        delete[] context->output;
#if not(DEBUG_USE_MMAP_FOR_WEIGHTS)
        if (context->weightSize.nBytes > 0)
            releaseAlignedBuffer(context->weight);
#endif
    }
    delete[] opForward;
    delete[] opContexts;
}

std::vector<NnByte *> NnCpuDevice::resolvePointer(NnSize3D *pntrSize, NnPointerConfig *pointerConfig) {
    NnByte *source;
    NnSize3D *sourceSize;

    switch (pointerConfig->source) {
    case SRC_BUFFER:
        source = buffers[pointerConfig->pointerIndex];
        sourceSize = &nodeConfig->buffers[pointerConfig->pointerIndex].size;
        break;
    case SRC_PIPE:
        source = netExecution->pipes[pointerConfig->pointerIndex];
        sourceSize = &netConfig->pipes[pointerConfig->pointerIndex].size;
        break;
    default:
        throw std::invalid_argument("Unsupported pointer type");
    }

    switch (pointerConfig->type) {
    case PNTR_RAW: {
        *pntrSize = size1D(sourceSize->floatType, sourceSize->length);
        return std::vector<NnByte *>{source};
    }
    case PNTR_BATCH:
    case PNTR_BATCHED_SLICE: {
        ASSERT_EQ(sourceSize->y, netConfig->nBatches);
        std::vector<NnByte *> pntr(sourceSize->z * sourceSize->y);

        NnSize batchBytes = getBytes(sourceSize->floatType, sourceSize->x);
        for (NnUint z = 0u; z < sourceSize->z; z++) {
            for (NnUint y = 0u; y < sourceSize->y; y++)
                pntr[z * sourceSize->y + y] = &source[(z * sourceSize->y + y) * batchBytes];
        }
        *pntrSize = *sourceSize;

        if (pointerConfig->type == PNTR_BATCHED_SLICE) {
            assert(sourceSize->x % netConfig->nNodes == 0);
            NnUint xSlice = sourceSize->x / netConfig->nNodes;
            NnSize xSliceBytes = getBytes(sourceSize->floatType, xSlice);
            for (NnUint z = 0; z < sourceSize->z; z++) {
                for (NnUint y = 0; y < sourceSize->y; y++)
                    pntr[z * sourceSize->y + y] = &pntr[z * sourceSize->y + y][xSliceBytes * nodeConfig->nodeIndex];
            }
            *pntrSize = size3D(sourceSize->floatType, sourceSize->z, sourceSize->y, xSlice);
        }
        return pntr;
    }
    default:
        throw std::invalid_argument("Unsupported pointer config");
    }
}

void NnCpuDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex >= 0u);
    assert(opIndex < nOps);
    NnCpuOpContext *context = &opContexts[opIndex];
    assert(offset + nBytes <= context->weightSize.nBytes);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(offset == 0u);
    context->weight = weight;
#else
    std::memcpy(&context->weight[offset], weight, nBytes);
#endif
}

void NnCpuDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) {
    NnCpuOpContext *context = &opContexts[opIndex];
    // printf("forward: %d %s (%d/%d)\n", opIndex, context->name, threadIndex + 1, nThreads); fflush(stdout);
    opForward[opIndex](nThreads, threadIndex, batchSize, context);
}
