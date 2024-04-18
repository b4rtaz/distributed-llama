#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"
#include "mixtral-tasks.hpp"

static TaskLoopTask inferenceTasks[] = {
    { llamaRmsAtt, TASK_TYPE_INFERENCE },
    { llamaRmsAttNorm, TASK_TYPE_INFERENCE },
    { llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE },
    { llamaSyncRmsAtt, TASK_TYPE_TRANSFER },
    { llamaQkv, TASK_TYPE_INFERENCE },
    { llamaQuantizeQkv, TASK_TYPE_INFERENCE },
    { llamaSyncQkv, TASK_TYPE_TRANSFER },
    { llamaDequantizeQkv, TASK_TYPE_INFERENCE },
    { llamaMultiheadAtt, TASK_TYPE_INFERENCE },
    { grokMultiheadAttRope, TASK_TYPE_INFERENCE },
    { llamaMultiheadAttJoin, TASK_TYPE_INFERENCE },
    { llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE },
    { llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER },
    { llamaAtt, TASK_TYPE_INFERENCE },
    { llamaQuantizeAtt, TASK_TYPE_INFERENCE },
    { llamaSyncAtt, TASK_TYPE_TRANSFER },
    { llamaDequantizeAtt, TASK_TYPE_INFERENCE },
    { llamaRmfFfn, TASK_TYPE_INFERENCE },
    { llamaRmfFfnNorm, TASK_TYPE_INFERENCE },

    { grokMoeRouter, TASK_TYPE_INFERENCE },
    { grokMoeRouterSoftmax, TASK_TYPE_INFERENCE },
    { grokMoeTopk, TASK_TYPE_INFERENCE },
    { grokMoeNormWeights, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeInput, TASK_TYPE_INFERENCE },
    { grokSyncMoeInput, TASK_TYPE_TRANSFER },
    { grokMoeBlock0, TASK_TYPE_INFERENCE },
    { grokMoeBlock1, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeMul, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulA, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulRearrange, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulB, TASK_TYPE_INFERENCE },
    { grokMoeBlock2, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeOutput, TASK_TYPE_INFERENCE },
    { grokSyncMoeOutput, TASK_TYPE_TRANSFER },
    { grokDequantizeMoeOutput, TASK_TYPE_INFERENCE },
    { grokMoeAdd, TASK_TYPE_INFERENCE },

    { llamaNextBlock, TASK_TYPE_INFERENCE },
    { llamaRmsFinal, TASK_TYPE_INFERENCE },
    { llamaRmsFinalNorm, TASK_TYPE_INFERENCE },
    { llamaFinalize, TASK_TYPE_INFERENCE },
};

static TaskLoopTask workerTasks[] = {
    { llamaSyncRmsAtt, TASK_TYPE_TRANSFER },
    { llamaQkv, TASK_TYPE_INFERENCE },
    { llamaQuantizeQkv, TASK_TYPE_INFERENCE },
    { llamaSyncQkv, TASK_TYPE_TRANSFER },
    { llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER },
    { llamaAtt, TASK_TYPE_INFERENCE },
    { llamaQuantizeAtt, TASK_TYPE_INFERENCE },
    { llamaSyncAtt, TASK_TYPE_TRANSFER },

    { grokSyncMoeInput, TASK_TYPE_TRANSFER },
    { grokMoeBlock0, TASK_TYPE_INFERENCE },
    { grokMoeBlock1, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeMul, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulA, TASK_TYPE_INFERENCE },
    { grokSyncMoeMulB, TASK_TYPE_INFERENCE },
    { grokMoeBlock2, TASK_TYPE_INFERENCE },
    { grokQuantizeMoeOutput, TASK_TYPE_INFERENCE },
    { grokSyncMoeOutput, TASK_TYPE_TRANSFER },

    { llamaNextBlock, TASK_TYPE_INFERENCE }
};

TransformerArch Mixtral::arch = {
    .initInference = NULL,
    .inference = {
        .nTasks = sizeof(inferenceTasks) / sizeof(TaskLoopTask),
        .tasks = inferenceTasks
    },
    .worker = {
        .nTasks = sizeof(workerTasks) / sizeof(TaskLoopTask),
        .tasks = workerTasks
    }
};
