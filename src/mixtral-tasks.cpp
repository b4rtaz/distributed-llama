#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"
#include "mixtral-tasks.hpp"

TransformerArch buildMixtralArch(TransformerSpec* spec) {
    TransformerArch a;

    // inference

    a.I(sendPoke, TASK_TYPE_TRANSFER);
    for (int i = 0; i < spec->nLayers; i++) {
        a.I(llamaRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaRmsAttNorm, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);
        a.I(llamaAttQ, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAttQ, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAttQ, TASK_TYPE_TRANSFER);
        a.I(llamaAttK, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAttK, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAttK, TASK_TYPE_TRANSFER);
        a.I(llamaAttV, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeAttV, TASK_TYPE_INFERENCE);
        a.I(llamaSyncAttV, TASK_TYPE_TRANSFER);
        a.I(llamaDequantizeQkv, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(grokMultiheadAttRope, TASK_TYPE_INFERENCE);
        a.I(llamaMultiheadAttJoin, TASK_TYPE_INFERENCE);
        a.I(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);
        a.I(llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER);
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
        a.W(llamaAttQ, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAttQ, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAttQ, TASK_TYPE_INFERENCE);
        a.W(llamaAttK, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAttK, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAttK, TASK_TYPE_INFERENCE);
        a.W(llamaAttV, TASK_TYPE_INFERENCE);
        a.W(llamaQuantizeAttV, TASK_TYPE_INFERENCE);
        a.W(llamaSyncAttV, TASK_TYPE_INFERENCE);
        a.W(llamaSyncMultiheadAtt, TASK_TYPE_TRANSFER);
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
