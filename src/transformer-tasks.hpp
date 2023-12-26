#ifndef TRANSFORMER_TASKS_HPP
#define TRANSFORMER_TASKS_HPP

#include "utils.hpp"
#include "transformer.hpp"

struct TransformerContext {
    Transformer* transformer;
    SocketPool* socketPool;
    bool finalize;
    unsigned int currentBlockIndex;
};

class Inference {
private:
    Transformer* transformer;
    TransformerContext context;
    TaskLoop *taskLoop;
public:
    static TaskLoopTask* tasks;
    static int nTasks;

    Inference(unsigned int nThreads, Transformer* transformer, SocketPool* socketPool);
    ~Inference();
    float* infer(int token, int pos);
};

class Worker {
private:
    Transformer* transformer;
    TransformerContext context;
    TaskLoop *taskLoop;
public:
    static TaskLoopTask* tasks;
    static int nTasks;

    Worker(unsigned int nThreads, Transformer* transformer, Socket* socket);
    ~Worker();
    void work();
};

#endif