#ifndef TRANSFORMER_TASKS_HPP
#define TRANSFORMER_TASKS_HPP

#include "utils.hpp"
#include "transformer.hpp"

struct TransformerContext {
    Transformer* transformer;
    int currentBlockIndex;
};

class Inference {
private:
    unsigned int nThreads;
    Transformer* transformer;
public:
    static TaskLoopTask* tasks;
    static int nTasks;

    Inference(unsigned int nThreads, Transformer* transformer);
    float* infer(int token, int pos);
};

#endif