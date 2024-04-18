#ifndef GROK_TASKS_HPP
#define GROK_TASKS_HPP

#include "tasks.hpp"

int grokMultiheadAttRope(TASK_ARGS);
int grokRmfFfn(TASK_ARGS);
int grokRmfFfnNorm(TASK_ARGS);
int grokRmfFfnNormJoin(TASK_ARGS);
int grokMoeRms(TASK_ARGS);
int grokMoeRmsNorm(TASK_ARGS);
int grokMoeRouter(TASK_ARGS);
int grokMoeRouterSoftmax(TASK_ARGS);
int grokMoeTopk(TASK_ARGS);
int grokMoeNormWeights(TASK_ARGS);
int grokQuantizeMoeInput(TASK_ARGS);
int grokSyncMoeInput(TASK_ARGS);
int grokMoeBlock0(TASK_ARGS);
int grokMoeBlock1(TASK_ARGS);
int grokQuantizeMoeMul(TASK_ARGS);
int grokSyncMoeMulA(TASK_ARGS);
int grokSyncMoeMulRearrange(TASK_ARGS);
int grokSyncMoeMulB(TASK_ARGS);
int grokMoeBlock2(TASK_ARGS);
int grokQuantizeMoeOutput(TASK_ARGS);
int grokSyncMoeOutput(TASK_ARGS);
int grokDequantizeMoeOutput(TASK_ARGS);
int grokMoeRmsFinal(TASK_ARGS);
int grokMoeRmsNormFinal(TASK_ARGS);
int grokMoeAdd(TASK_ARGS);

class Grok1 {
    public:
        static TransformerArch arch;
};

#endif