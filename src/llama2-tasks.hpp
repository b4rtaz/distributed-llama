#ifndef LLAMA2_TASKS_HPP
#define LLAMA2_TASKS_HPP

#include "tasks.hpp"

int llamaRmsAtt(TASK_ARGS);
int llamaRmsAttNorm(TASK_ARGS);
int llamaQuantizeRmsAtt(TASK_ARGS);
int llamaSyncRmsAtt(TASK_ARGS);
int llamaQkv(TASK_ARGS);
int llamaQuantizeQkv(TASK_ARGS);
int llamaSyncQkv(TASK_ARGS);
int llamaDequantizeQkv(TASK_ARGS);
int llamaMultiheadAtt(TASK_ARGS);
int llamaMultiheadAttRope(TASK_ARGS);
int llamaMultiheadAttJoin(TASK_ARGS);
int llamaQuantizeMultiheadAtt(TASK_ARGS);
int llamaSyncMultiheadAtt(TASK_ARGS);
int llamaAtt(TASK_ARGS);
int llamaQuantizeAtt(TASK_ARGS);
int llamaSyncAtt(TASK_ARGS);
int llamaDequantizeAtt(TASK_ARGS);
int llamaRmfFfn(TASK_ARGS);
int llamaRmfFfnNorm(TASK_ARGS);
int llamaNextBlock(TASK_ARGS);
int llamaRmsFinal(TASK_ARGS);
int llamaRmsFinalNorm(TASK_ARGS);
int llamaFinalize(TASK_ARGS);

class Llama2 {
    public:
        static TransformerArch arch;
};

#endif