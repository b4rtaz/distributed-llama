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
    // handler --> sendPos | llamaRmsAtt ... --> 指定任务处理函数
    // taskType: inference --> 0, transfer --> 1
    // tasks --> 指向TransformerTasks结构的指针,表示传输的任务队列
    const int alloc = 32; // 内存分配时的增量大小
    if (tasks->nTasks % alloc == 0) { // tasks->nTasks在逐渐增大,每增大32个,则触发一次内存分配
        /*
        typedef struct { // --> 任务处理程序和任务类型
            TaskLoopHandler* handler;
            unsigned int taskType;
        } TaskLoopTask;
        */
        TaskLoopTask* newTasks = new TaskLoopTask[tasks->nTasks + alloc];
        if (tasks->nTasks > 0) { // 如果原本的tasks->nTasks中仍然存在任务
            memcpy(newTasks, tasks->tasks, tasks->nTasks * sizeof(TaskLoopTask));// 把未完成的任务cp到newTasks中
            delete[] tasks->tasks; // 删除空间
        }
        tasks->tasks = newTasks;
    }
    // 每32个任务触发一次内存分配
    tasks->tasks[tasks->nTasks].handler = handler;
    tasks->tasks[tasks->nTasks].taskType = taskType;
    tasks->nTasks++; // 更新task的数量
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
    // printf("\n syncUnitBuffer:%ld \n", bufferBytes);
    // printf("\n syncUnitBuffer bufferBytes %ld\n", bufferBytes);

    if (ctx->socketPool != NULL) { // 通过ctx->scoketPool是否是NULL来判断是ROOT Node还是Worker Node
        // root
        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        SocketIo ios[nSockets];
        for (int i = 0; i < nSockets; i++) {
            // printf("\n fasong %d ci\n", i);
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = buffer;
            ios[i].size = bufferBytes;
        }
        ctx->socketPool->writeMany(nSockets, ios); // 从ROOT Node发送数据到Worker Node
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return; // 对于Worker Node,只有线程0进行数据接收

        // worker
        ctx->socket->read(buffer, bufferBytes);
    }
}

void syncSliceOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    int sumMemoryBugdet = 0;
    for (int value:ctx->transformer->spec->memoryBudgetArray){
        sumMemoryBugdet += value;
    }
    size_t bufferBytes = ctx->transformer->buffer->getSlicedBytes(bufferIndex, sumMemoryBugdet, false); // 从TB_SLICED_XBV_QUANTIZED取出量化之后的注意力输出结果,ROOT Node进行发送,Worker Node进行接收
    // size_t bufferBytes = 8192;
    // 这里的bufferBytes是没有 * memoryBudget的
    // printf("\n syncSliceOfSlicedBuffer bufferBytes %ld\n", bufferBytes);
    if (ctx->socketPool != NULL) {
        // root
        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        SocketIo ios[nSockets];
        for (int i = 0; i < nSockets; i++) {
            int socketIndex = threadIndex + i * nThreads;
            uint8_t workerSliceIndex = socketIndex + 1;
            int midSumMemoryBudget = 0;
            for (int i = 0; i < workerSliceIndex; i++){
                midSumMemoryBudget += ctx->transformer->spec->memoryBudgetArray[i];
            }
            int memoryBudget = ctx->transformer->spec->memoryBudgetArray[workerSliceIndex];
            ios[i].socketIndex = socketIndex;
            ios[i].data = ctx->transformer->buffer->getSliced(bufferIndex, workerSliceIndex, midSumMemoryBudget, sumMemoryBugdet, false);
            // 问题在这里的getSliced,应该是2048结果是3072!!!
            ios[i].size = bufferBytes;
            // printf("\n syncSliceOfSlicedBuffer Root Read Size:%ld \n", ios[i].size);
        }

        ctx->socketPool->readMany(nSockets, ios);
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return;
        int midSumMemoryBudget = 0;
        for (int i = 0; i < ctx->transformer->sliceIndex; i++){
            midSumMemoryBudget += ctx->transformer->spec->memoryBudgetArray[i];
        }
        // worker
        int memoryBudget = ctx->transformer->spec->memoryBudgetArray[ctx->transformer->sliceIndex];
        void* buffer = ctx->transformer->buffer->getSliced(bufferIndex, ctx->transformer->sliceIndex, midSumMemoryBudget, sumMemoryBugdet, true);
        // float* test = (float *)buffer;
        // for (int i = 0; i < 5000; i++){
        //     printf("\n syncSliceOfSlicedBuffer i:%d \t Write:%f\n", i, test[i]);
        // }
        // for (int im syncSliceOfSlicedBuffer Worker Write Size:%ld \n", bufferBytes * sumMemoryBugdet);
        ctx->socket->write(buffer, bufferBytes);
    }
}

void syncMissingSlicesOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex) {
    int sumMemoryBugdet = 0;
    for (int value:ctx->transformer->spec->memoryBudgetArray){
        sumMemoryBugdet += value;
    }
    size_t sliceBytes = ctx->transformer->buffer->getSlicedBytes(bufferIndex, sumMemoryBugdet, true);
    if (ctx->socketPool != NULL) {
        // root

        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        SocketIo ios[nSockets];

        for (uint8_t si = 0; si < ctx->transformer->spec->nSlices - 1; si++) {
            for (unsigned int i = 0; i < nSockets; i++) {
                int socketIndex = threadIndex + i * nThreads;
                uint8_t workerSliceIndex = socketIndex + 1;
                slice_index_t sliceIndex = si < workerSliceIndex ? si : si + 1;
                ios[i].socketIndex = socketIndex;
                int midSumMemoryBudget = 0;
                for (int i = 0; i < sliceIndex; i++){
                    midSumMemoryBudget += ctx->transformer->spec->memoryBudgetArray[i];
                }
                int memoryBudget = ctx->transformer->spec->memoryBudgetArray[sliceIndex];
                ios[i].data = ctx->transformer->buffer->getSliced(bufferIndex, sliceIndex, midSumMemoryBudget, sumMemoryBugdet, true);
                ios[i].size = sliceBytes * memoryBudget;
            }
            ctx->socketPool->writeMany(nSockets, ios);
        }
    } else if (ctx->socket != NULL) {
        if (threadIndex != 0) return;

        // worker
        for (slice_index_t sliceIndex = 0; sliceIndex < ctx->transformer->spec->nSlices; sliceIndex++) {
            if (sliceIndex != ctx->transformer->sliceIndex) {
                int memoryBudget = ctx->transformer->spec->memoryBudgetArray[sliceIndex];
                int midSumMemoryBudget = 0;
                for (int i = 0; i < sliceIndex; i++){
                    midSumMemoryBudget += ctx->transformer->spec->memoryBudgetArray[i];
                }
                void* buffer = ctx->transformer->buffer->getSliced(bufferIndex, sliceIndex, midSumMemoryBudget, sumMemoryBugdet, true);
                ctx->socket->read(buffer, sliceBytes * memoryBudget);
            }
        }
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
    int sumMemoryBugdet = 0;
    for (int value:ctx->transformer->spec->memoryBudgetArray){
        sumMemoryBugdet += value;
    }
    int midSumMemoryBudget = 0;
    for (int i = 0; i < ctx->transformer->sliceIndex; i++){
        midSumMemoryBudget += ctx->transformer->spec->memoryBudgetArray[i];
    }
    int memoryBudget = ctx->transformer->spec->memoryBudgetArray[ctx->transformer->sliceIndex];
    quantizeQ80Row(
        (float*)ctx->transformer->buffer->getSliced(sourceBufferIndex, ctx->transformer->sliceIndex, midSumMemoryBudget, sumMemoryBugdet, true),
        (BlockQ80*)ctx->transformer->buffer->getSliced(targetBufferIndex, ctx->transformer->sliceIndex, midSumMemoryBudget, sumMemoryBugdet, true),
        ctx->transformer->buffer->getSlicedBytes(sourceBufferIndex, sumMemoryBugdet, true) / sizeof(float),
        nThreads,
        threadIndex);
}

void dequantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool dequantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex) {
    if (ctx->transformer->spec->bufferFloatType == F32) return;
    assert(ctx->transformer->spec->bufferFloatType == Q80);
    assert(ctx->socketPool != NULL); // This function may be called only by root.

    unsigned int sliceIndex = dequantizeRootSlice ? 0 : 1;
    for (; sliceIndex < ctx->transformer->spec->nSlices; sliceIndex++) {
        int sumMemoryBugdet = 0;
        for (int value:ctx->transformer->spec->memoryBudgetArray){
            sumMemoryBugdet += value;
        }
        int midSumMemoryBudget = 0;
        for (int i = 0; i < sliceIndex; i++){
            midSumMemoryBudget += ctx->transformer->spec->memoryBudgetArray[i];
        }
        int memoryBudget = ctx->transformer->spec->memoryBudgetArray[sliceIndex];
        dequantizeQ80Row(
            (BlockQ80*)ctx->transformer->buffer->getSliced(sourceBufferIndex, sliceIndex, midSumMemoryBudget, sumMemoryBugdet, true),
            (float*)ctx->transformer->buffer->getSliced(targetBufferIndex, sliceIndex, midSumMemoryBudget, sumMemoryBugdet, true),
            (ctx->transformer->buffer->getSlicedBytes(sourceBufferIndex, sumMemoryBugdet, true) / sizeof(BlockQ80)) * QK80,
            nThreads,
            threadIndex);
    }
}

void sendPos(TASK_ARGS) {
    // TASK_ARGS --> unsigned int nThreads, unsigned int threadIndex, void* userData
    TASK_VARIABLES;
    /* TASK_VARIABLES --> 
        TransformerContext* ctx = (TransformerContext*)userData; 作为参数传入的
        Transformer* transformer = ctx->transformer;
        TransformerBlock* block = transformer->blocks[ctx->currentBlockIndex];
        TransformerSpec* spec = transformer->spec;
    */

    if (ctx->socketPool != NULL) {
        // 把socket分配给每个进程
        // 这里的nSockets代表当前线程需要处理多少个socket进程

        unsigned int nSockets = ctx->socketPool->nSockets / nThreads + (ctx->socketPool->nSockets % nThreads > threadIndex ? 1 : 0);
        
        SocketIo ios[nSockets];
        for (int i = 0; i < nSockets; i++) {
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = &transformer->pos;
            ios[i].size = sizeof(pos_t);
        }
        ctx->socketPool->writeMany(nSockets, ios); // 发送数据,需要发送的数据就是transformer->pos中的数据,因此ios[i].data是一个指针,指向了transformer->pos的地址
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

    // contentRow是输入token在词嵌入表中的起始地址
    float* contentRow = ((float*)transformer->tokenEmbeddingTable) + token * transformer->spec->dim;
    // (float*)transformer->tokenEmbeddingTable得到的是词嵌入表的起始地址,
    // token是这个输入词的索引,transformer->spec->dim是一个token的维度,两者相乘得到这个输入词在嵌入表中的位置

    memcpy(transformer->x, contentRow, transformer->spec->dim * sizeof(float));

    context.currentBlockIndex = 0; // Block Index
    taskLoop->run(); // 创建多线程,循环执行任务
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
