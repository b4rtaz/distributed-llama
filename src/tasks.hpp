#ifndef TASKS_HPP
#define TASKS_HPP

#include "transformer.hpp"
#include "utils.hpp"

#define TASK_ARGS unsigned int nThreads, unsigned int threadIndex, void* userData

#define TASK_N_TYPES 2
#define TASK_TYPE_INFERENCE 0
#define TASK_TYPE_TRANSFER 1

struct TransformerContext {
    Transformer* transformer;
    Socket* socket;
    SocketPool* socketPool;
    unsigned int currentBlockIndex;
};

typedef void (InferenceInitializer)(TransformerContext* context);

struct TransformerTasks {
    unsigned int nTasks;
    TaskLoopTask* tasks;
};

class TransformerArch {
public:
    TransformerTasks inference;
    TransformerTasks worker;

    TransformerArch();
    ~TransformerArch();

    void I(TaskLoopHandler* handler, unsigned int taskType);
    void W(TaskLoopHandler* handler, unsigned int taskType);
};

#define TASK_VARIABLES \
    TransformerContext* ctx = (TransformerContext*)userData; \
    Transformer* transformer = ctx->transformer; \
    TransformerBlock* block = transformer->blocks[ctx->currentBlockIndex]; \
    TransformerSpec* spec = transformer->spec; // printf("%s:%d\n", __FUNCTION__, ctx->currentBlockIndex); fflush(stdout);

void syncUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex);
void syncSliceOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex);
void quantizeUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t sourceBufferIndex, uint8_t targetBufferIndex);
void quantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool quantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex);
void dequantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool dequantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex);
void sendPos(TASK_ARGS);

class Inference {
private:
    Transformer* transformer;
    SocketPool* socketPool;
    TransformerContext context;
    TaskLoop *taskLoop;
    TransformerArch *arch;
public:
    Inference(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, SocketPool* socketPool);
    ~Inference();
    float* infer(int token, pos_t pos);
    void getStats(unsigned long* inferenceTime, unsigned long* transferTime);
};

class Worker {
private:
    Transformer* transformer;
    Socket* socket;
    TransformerContext context;
    TaskLoop *taskLoop;
public:
    Worker(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, Socket* socket);
    ~Worker();
    void work();
};

#endif
