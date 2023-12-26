#ifndef TRANSFORMER_TASKS_HPP
#define TRANSFORMER_TASKS_HPP

#include "utils.hpp"
#include "transformer.hpp"

struct TransformerContext {
    Transformer* transformer;
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

    Inference(unsigned int nThreads, Transformer* transformer);
    ~Inference();
    float* infer(int token, int pos);
};

#endif