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

    if (ctx->socketPool != NULL) {
        // root

        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSockets > 0) {
            SocketIo ios[nSockets];
            for (int i = 0; i < nSockets; i++) {
                ios[i].socketIndex = threadIndex + i * nThreads;
                ios[i].data = buffer;
                ios[i].size = bufferBytes;
            }
            ctx->socketPool->writeMany(nSockets, ios);
        }
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return;

        // worker
        ctx->socket->read(buffer, bufferBytes);
    }
}

void syncSliceOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    size_t bufferBytes = ctx->transformer->buffer->getSlicedBytes(bufferIndex);
    if (ctx->socketPool != NULL) {
        // root

        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSockets > 0) {
            SocketIo ios[nSockets];
            for (int i = 0; i < nSockets; i++) {
                int socketIndex = threadIndex + i * nThreads;
                uint8_t workerSliceIndex = socketIndex + 1;
                ios[i].socketIndex = socketIndex;
                ios[i].data = ctx->transformer->buffer->getSliced(bufferIndex, workerSliceIndex);
                ios[i].size = bufferBytes;
            }

            ctx->socketPool->readMany(nSockets, ios);
        }
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return;

        // worker
        void* buffer = ctx->transformer->buffer->getSliced(bufferIndex, ctx->transformer->sliceIndex);
        ctx->socket->write(buffer, bufferBytes);
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

void quantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool quantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    if (ctx->transformer->sliceIndex == 0 && !quantizeRootSlice) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);

    quantizeQ80Row(
        (float*)ctx->transformer->buffer->getSliced(sourceBufferIndex, ctx->transformer->sliceIndex),
        (BlockQ80*)ctx->transformer->buffer->getSliced(targetBufferIndex, ctx->transformer->sliceIndex),
        ctx->transformer->buffer->getSlicedBytes(sourceBufferIndex) / sizeof(float),
        nThreads,
        threadIndex);
}

void dequantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool dequantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);
    assert(ctx->socketPool != NULL); // This function may be called only by root.

    unsigned int sliceIndex = dequantizeRootSlice ? 0 : 1;
    for (; sliceIndex < ctx->transformer->spec->nSlices; sliceIndex++) {
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

    if (ctx->socketPool != NULL) {
        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSockets > 0) {
            SocketIo ios[nSockets];
            for (int i = 0; i < nSockets; i++) {
                ios[i].socketIndex = threadIndex + i * nThreads;
                ios[i].data = &transformer->pos;
                ios[i].size = sizeof(pos_t);
            }
            ctx->socketPool->writeMany(nSockets, ios);
        }
    }
}

bool tryWaitForPos(Transformer* transformer, Socket* socket, unsigned int maxAttempts) {
    return socket->tryRead(&transformer->pos, sizeof(pos_t), maxAttempts);
}

Inference::Inference(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, SocketPool* socketPool) {
    this->transformer = transformer;
    this->socketPool = socketPool;
    this->arch = arch;
    context.transformer = transformer;
    context.socket = NULL;
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
    memcpy(transformer->x, contentRow, transformer->spec->dim * sizeof(float));

    context.currentBlockIndex = 0;

    taskLoop->run();

    return transformer->logits;
}

void Inference::getStats(unsigned long* inferenceTime, unsigned long* transferTime) {
    *inferenceTime = taskLoop->executionTime[TASK_TYPE_INFERENCE];
    *transferTime = taskLoop->executionTime[TASK_TYPE_TRANSFER];
}

Worker::Worker(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, Socket* socket) {
    this->transformer = transformer;
    this->socket = socket;
    context.transformer = transformer;
    context.socket = socket;
    context.socketPool = NULL;
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

        while (!tryWaitForPos(transformer, socket, maxAttempts)) {
            if (turbo) {
                // After one second of waiting with non-blocking read, we switch to blocking mode to not burn CPU.
                if (clock() - start > CLOCKS_PER_SEC) {
                    socket->setTurbo(false);
                    turbo = false;
                    printf("🚁 Socket is in blocking mode\n");
                }
            }
        }
        if (!turbo) {
            socket->setTurbo(true);
            turbo = true;
            printf("🚁 Socket is in non-blocking mode\n");
        }

        context.currentBlockIndex = 0;
        taskLoop->run();
    }
}
