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
    bool finalize;
    unsigned int currentBlockIndex;
};

typedef void (InferenceInitializer)(TransformerContext* context);

struct TransformerTasks {
    unsigned int nTasks;
    TaskLoopTask* tasks;
};


struct TransformerArch {
    InferenceInitializer* initInference;
    TransformerTasks inference;
    TransformerTasks worker;
};

#define TASK_VARIABLES \
    TransformerContext* ctx = (TransformerContext*)userData; \
    Transformer* transformer = ctx->transformer; \
    TransformerBlock* block = transformer->blocks[ctx->currentBlockIndex]; \
    TransformerSpec* spec = transformer->spec;

void syncUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex);
void syncSliceOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex);
void syncMissingSlicesOfSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t bufferIndex);
void quantizeUnitBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, uint8_t sourceBufferIndex, uint8_t targetBufferIndex);
void quantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool quantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex);
void dequantizeSlicedBuffer(unsigned int nThreads, unsigned int threadIndex, TransformerContext* ctx, bool dequantizeRootSlice, uint8_t sourceBufferIndex, uint8_t targetBufferIndex);

class Inference {
private:
    Transformer* transformer;
    TransformerContext context;
    TaskLoop *taskLoop;
    TransformerArch *arch;
public:
    Inference(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, SocketPool* socketPool);
    ~Inference();
    float* infer(int token, int pos);
    void getStats(unsigned long* inferenceTime, unsigned long* transferTime);
};

class Worker {
private:
    Transformer* transformer;
    TransformerContext context;
    TaskLoop *taskLoop;
public:
    Worker(TransformerArch* arch, unsigned int nThreads, Transformer* transformer, Socket* socket);
    ~Worker();
    void work();
};

#endif
