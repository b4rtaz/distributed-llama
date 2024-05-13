#ifndef GROK_TASKS_HPP
#define GROK_TASKS_HPP

#include "tasks.hpp"

void grokRmfFfn(TASK_ARGS);
void grokRmfFfnNorm(TASK_ARGS);
void grokRmfFfnNormJoin(TASK_ARGS);
void grokMoeRms(TASK_ARGS);
void grokMoeRmsNorm(TASK_ARGS);
void grokMoeRouter(TASK_ARGS);
void grokMoeRouterSoftmax(TASK_ARGS);
void grokMoeTopk(TASK_ARGS);
void grokMoeNormWeights(TASK_ARGS);
void grokQuantizeMoeInput(TASK_ARGS);
void grokSyncMoeInput(TASK_ARGS);
void grokMoeBlock0(TASK_ARGS);
void grokMoeBlock1(TASK_ARGS);
void grokQuantizeMoeMul(TASK_ARGS);
void grokSyncMoeMulA(TASK_ARGS);
void grokSyncMoeMulRearrange(TASK_ARGS);
void grokSyncMoeMulB(TASK_ARGS);
void grokMoeBlock2(TASK_ARGS);
void grokQuantizeMoeOutput(TASK_ARGS);
void grokSyncMoeOutput(TASK_ARGS);
void grokDequantizeMoeOutput(TASK_ARGS);
void grokMoeRmsFinal(TASK_ARGS);
void grokMoeRmsNormFinal(TASK_ARGS);
void grokMoeAdd(TASK_ARGS);

TransformerArch buildGrok1Arch(TransformerSpec* spec);

#endif