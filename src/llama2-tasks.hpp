#ifndef LLAMA2_TASKS_HPP
#define LLAMA2_TASKS_HPP

#include "tasks.hpp"

void llamaRmsAtt(TASK_ARGS);
void llamaRmsAttNorm(TASK_ARGS);
void llamaQuantizeRmsAtt(TASK_ARGS);
void llamaSyncRmsAtt(TASK_ARGS);
void llamaQkv(TASK_ARGS);
void llamaRope(TASK_ARGS);
void llamaMultiheadAtt(TASK_ARGS);
void llamaQuantizeMultiheadAtt(TASK_ARGS);
void llamaAtt(TASK_ARGS);
void llamaQuantizeAtt(TASK_ARGS);
void llamaSyncAtt(TASK_ARGS);
void llamaDequantizeAtt(TASK_ARGS);
void llamaMergeAtt(TASK_ARGS);
void llamaRmfFfn(TASK_ARGS);
void llamaRmfFfnNorm(TASK_ARGS);
void llamaNextBlock(TASK_ARGS);
void llamaRmsFinal(TASK_ARGS);
void llamaRmsFinalNorm(TASK_ARGS);
void llamaFinalize(TASK_ARGS);

TransformerArch buildLlamaArch(TransformerSpec* spec);

#endif