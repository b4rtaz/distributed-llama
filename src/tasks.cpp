#include <cassert>
#include <cstring>
#include <cstdio>
#include <ctime>
#include "tasks.hpp"

TransformerArch::TransformerArch() {
    inference.nTasks = 0;
    worker.nTasks = 0;
}

TransformerArch::~TransformerArch() {
    if (inference.nTasks > 0) {
        delete[] inference.tasks;
    }
    if (worker.nTasks > 0) {
        delete[] worker.tasks;
    }
}

void addTask(TaskLoopHandler* handler, unsigned int taskType, TransformerTasks* tasks) {
    const int alloc = 32;
    if (tasks->nTasks % alloc == 0) {
        TaskLoopTask* newTasks = new TaskLoopTask[tasks->nTasks + alloc];
        if (tasks->nTasks > 0) {
            memcpy(newTasks, tasks->tasks, tasks->nTasks * sizeof(TaskLoopTask));
            delete[] tasks->tasks;
        }
        tasks->tasks = newTasks;
    }
    tasks->tasks[tasks->nTasks].handler = handler;
    tasks->tasks[tasks->nTasks].taskType = taskType;
    tasks->nTasks++;
}

void TransformerArch::I(TaskLoopHandler* handler, unsigned int taskType) {
    addTask(handler, taskType, &inference);
}

void TransformerArch::W(TaskLoopHandler* handler, unsigned int taskType) {
    addTask(handler, taskType, &worker);
}

void syncUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    void* buffer = ctx->transformer->buffer->getUnit(bufferIndex);
    size_t bufferBytes = ctx->transformer->buffer->getUnitBytes(bufferIndex);

    if (ctx->transformer->sliceIndex == 0) {
        // root

        unsigned int nSocketsPerThread = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSocketsPerThread == 0) return;

        SocketIo ios[nSocketsPerThread];
        for (int i = 0; i < nSocketsPerThread; i++) {
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = buffer;
            ios[i].size = bufferBytes;
        }
        ctx->socketPool->writeManyWithAlignment(nSocketsPerThread, ios);
    } else {
        // worker

        if (threadIndex != 0) return;

        SocketIo ios;
        ios.data = buffer;
        ios.size = bufferBytes;
        ios.socketIndex = ROOT_SOCKET_INDEX;
        ctx->socketPool->readManyWithAlignment(1, &ios);
    }
}

void syncSliceOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, bool onlyFromWorkerToRoot, TransformerContext* ctx, uint8_t bufferIndex) {
    bool isWorker = ctx->transformer->sliceIndex != 0;
    unsigned int nSockets = onlyFromWorkerToRoot && isWorker ? 1 : ctx->socketPool->nSockets;
    unsigned int nSocketsPerThread = nSockets / nThreads + (nSockets % nThreads > threadIndex ? 1 : 0);
    if (nSocketsPerThread == 0) return;

    size_t sliceBytes = ctx->transformer->buffer->getSlicedBytes(bufferIndex);

    if (!onlyFromWorkerToRoot || isWorker) {
        void* mySliceData = ctx->transformer->buffer->getSliced(bufferIndex, ctx->transformer->sliceIndex);

        SocketIo writeIos[nSocketsPerThread];
        for (unsigned int i = 0; i < nSocketsPerThread; i++) {
            unsigned int socketIndex = threadIndex + i * nThreads;
            writeIos[i].socketIndex = socketIndex;
            writeIos[i].data = mySliceData;
            writeIos[i].size = sliceBytes;
        }
        ctx->socketPool->writeManyWithAlignment(nSocketsPerThread, writeIos);
    }

    if (!onlyFromWorkerToRoot || !isWorker) {
        SocketIo readIos[nSocketsPerThread];
        for (unsigned int i = 0; i < nSocketsPerThread; i++) {
            unsigned int socketIndex = threadIndex + i * nThreads;
            int sliceIndex = socketIndex >= ctx->transformer->sliceIndex ? socketIndex + 1 : socketIndex;
            readIos[i].socketIndex = socketIndex;
            readIos[i].data = ctx->transformer->buffer->getSliced(bufferIndex, sliceIndex);
            readIos[i].size = sliceBytes;
        }
        ctx->socketPool->readManyWithAlignment(nSocketsPerThread, readIos);
    }
}

void quantizeUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);

    quantizeQ80Row(
        (float*)ctx->transformer->buffer->getUnit(sourceBufferIndex),
        (BlockQ80*)ctx->transformer->buffer->getUnit(targetBufferIndex),
        ctx->transformer->buffer->getUnitBytes(sourceBufferIndex) / sizeof(float),
        nThreads,
        threadIndex);
}

void quantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);

    quantizeQ80Row(
        (float*)ctx->transformer->buffer->getSliced(sourceBufferIndex, ctx->transformer->sliceIndex),
        (BlockQ80*)ctx->transformer->buffer->getSliced(targetBufferIndex, ctx->transformer->sliceIndex),
        ctx->transformer->buffer->getSlicedBytes(sourceBufferIndex) / sizeof(float),
        nThreads,
        threadIndex);
}

void dequantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool skipMySlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);
    assert(ctx->socketPool != NULL); // This function may be called only by root.

    for (unsigned int sliceIndex = 0; sliceIndex < ctx->transformer->spec->nSlices; sliceIndex++) {
        if (skipMySlice && sliceIndex == ctx->transformer->sliceIndex)
            continue;
        dequantizeQ80Row(
            (BlockQ80*)ctx->transformer->buffer->getSliced(sourceBufferIndex, sliceIndex),
            (float*)ctx->transformer->buffer->getSliced(targetBufferIndex, sliceIndex),
            (ctx->transformer->buffer->getSlicedBytes(sourceBufferIndex) / sizeof(BlockQ80)) * QK80,
            nThreads,
            threadIndex);
    }
}

void sendPos(TASK_ARGS) {
    TASK_VARIABLES;
    unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
    if (nSockets > 0) {
        SocketIo ios[nSockets];
        for (int i = 0; i < nSockets; i++) {
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = &transformer->pos;
            ios[i].size = sizeof(pos_t);
        }
        ctx->socketPool->writeManyWithAlignment(nSockets, ios);
    }
}

bool tryWaitForPos(Transformer* transformer, SocketPool* socketPool, unsigned int maxAttempts) {
    return socketPool->tryReadWithAlignment(ROOT_SOCKET_INDEX, &transformer->pos, sizeof(pos_t), maxAttempts);
}

Inference::Inference(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, SocketPool* socketPool) {
    this->transformer = transformer;
    this->socketPool = socketPool;
    this->arch = arch;
    context.transformer = transformer;
    context.socketPool = socketPool;
    assert(arch->inference.tasks[0].handler == sendPos);
    taskLoop = new TaskLoop(nThreads, arch->inference.nTasks, TASK_N_TYPES, arch->inference.tasks, (void*)&context);
}

Inference::~Inference() {
    delete taskLoop;
}

float* Inference::infer(int token, pos_t pos) {
    transformer->pos = pos;

    float* contentRow = ((float*)transformer->tokenEmbeddingTable) + token * transformer->spec->dim;
    float* x = (float*)transformer->buffer->getUnit(TB_UNIT_X);
    memcpy(x, contentRow, transformer->spec->dim * sizeof(float));

    context.currentBlockIndex = 0;

    taskLoop->run();

    return (float*)transformer->buffer->getUnit(TB_SLICED_LOGITS);
}

void Inference::getStats(unsigned long* inferenceTime, unsigned long* transferTime) {
    *inferenceTime = taskLoop->executionTime[TASK_TYPE_INFERENCE];
    *transferTime = taskLoop->executionTime[TASK_TYPE_TRANSFER];
}

Worker::Worker(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, SocketPool* socketPool) {
    this->transformer = transformer;
    this->socketPool = socketPool;
    context.transformer = transformer;
    context.socketPool = socketPool;
    taskLoop = new TaskLoop(nThreads, arch->worker.nTasks, TASK_N_TYPES, arch->worker.tasks, (void*)&context);
}

Worker::~Worker() {
    delete taskLoop;
}

void Worker::work() {
    const unsigned long maxAttempts = 10000;

    bool turbo = false;
    while (true) {
        const clock_t start = clock();

        while (!tryWaitForPos(transformer, socketPool, maxAttempts)) {
            if (turbo) {
                // After one second of waiting with non-blocking read, we switch to blocking mode to not burn CPU.
                if (clock() - start > CLOCKS_PER_SEC) {
                    socketPool->setTurbo(false);
                    turbo = false;
                    printf("🚁 Socket is in blocking mode\n");
                }
            }
        }
        if (!turbo) {
            socketPool->setTurbo(true);
            turbo = true;
            printf("🚁 Socket is in non-blocking mode\n");
        }

        context.currentBlockIndex = 0;
        taskLoop->run();

        unsigned int inferenceTime = taskLoop->executionTime[TASK_TYPE_INFERENCE];
        unsigned int transferTime = taskLoop->executionTime[TASK_TYPE_TRANSFER];
        printf("🔶 I %4u ms T %4u ms\n", inferenceTime, transferTime);
    }
}
