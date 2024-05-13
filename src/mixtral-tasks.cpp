#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"
#include "mixtral-tasks.hpp"

TransformerArch buildMixtralArch(TransformerSpec* spec) {
    TransformerArch a;

    // inference

    a.I(sendPos, TASK_TYPE_TRANSFER);
    for (int i = 0; i < spec->nLayers; i++) {
        a.I(llamaRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaRmsAttNorm, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.I(llamaQkv, TASK_TYPE_INFERENCE);
        a.I(llamaRope, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(llamaAtt, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAtt, TASK_TYPE_TRANSFER);
        a.I(llamaDequantizeAtt, TASK_TYPE_INFERENCE);
        a.I(llamaMergeAtt, TASK_TYPE_INFERENCE);
        a.I(llamaRmfFfn, TASK_TYPE_INFERENCE);
        a.I(llamaRmfFfnNorm, TASK_TYPE_INFERENCE);

        a.I(grokMoeRouter, TASK_TYPE_INFERENCE);
        a.I(grokMoeRouterSoftmax, TASK_TYPE_INFERENCE);
        a.I(grokMoeTopk, TASK_TYPE_INFERENCE);
        a.I(grokMoeNormWeights, TASK_TYPE_INFERENCE);
        a.I(grokQuantizeMoeInput, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeInput, TASK_TYPE_TRANSFER);
        a.I(grokMoeBlock0, TASK_TYPE_INFERENCE);
        a.I(grokMoeBlock1, TASK_TYPE_INFERENCE);
        a.I(grokQuantizeMoeMul, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeMulA, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeMulRearrange, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeMulB, TASK_TYPE_INFERENCE);
        a.I(grokMoeBlock2, TASK_TYPE_INFERENCE);
        a.I(grokQuantizeMoeOutput, TASK_TYPE_INFERENCE);
        a.I(grokSyncMoeOutput, TASK_TYPE_TRANSFER);
        a.I(grokDequantizeMoeOutput, TASK_TYPE_INFERENCE);
        a.I(grokMoeAdd, TASK_TYPE_INFERENCE);

        a.I(llamaNextBlock, TASK_TYPE_INFERENCE);
    }
    a.I(llamaRmsFinal, TASK_TYPE_INFERENCE);
    a.I(llamaRmsFinalNorm, TASK_TYPE_INFERENCE);
    a.I(llamaFinalize, TASK_TYPE_INFERENCE);

    // worker

    for (int i = 0; i < spec->nLayers; i++) {
        a.W(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.W(llamaQkv, TASK_TYPE_INFERENCE);
        a.W(llamaRope, TASK_TYPE_INFERENCE);
        a.W(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);
        a.W(llamaAtt, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAtt, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAtt, TASK_TYPE_TRANSFER);

        a.W(grokSyncMoeInput, TASK_TYPE_TRANSFER);
        a.W(grokMoeBlock0, TASK_TYPE_INFERENCE);
        a.W(grokMoeBlock1, TASK_TYPE_INFERENCE);
        a.W(grokQuantizeMoeMul, TASK_TYPE_INFERENCE);
        a.W(grokSyncMoeMulA, TASK_TYPE_INFERENCE);
        a.W(grokSyncMoeMulB, TASK_TYPE_INFERENCE);
        a.W(grokMoeBlock2, TASK_TYPE_INFERENCE);
        a.W(grokQuantizeMoeOutput, TASK_TYPE_INFERENCE);
        a.W(grokSyncMoeOutput, TASK_TYPE_TRANSFER);

        a.W(llamaNextBlock, TASK_TYPE_INFERENCE);
    }

    return a;
}
