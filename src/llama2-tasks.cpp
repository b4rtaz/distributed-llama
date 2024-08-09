#include <cmath>
#include <cassert>
#include <string.h>
#include "utils.hpp"
#include "funcs.hpp"
#include "socket.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"

void llamaRmsAtt(TASK_ARGS) { // 仅在线程0中执行,计算Transformer的block的RMS值
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
}

void llamaRmsAttNorm(TASK_ARGS) { // 通过上面计算的rms快速计算归一化值并存储在TB_UNIT_XB中
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    rmsnorm(xb, transformer->x, transformer->rms, block->rmsAtt, spec->dim, nThreads, threadIndex);
}
// --------------------------------------------------------------------------------------------------------------------------------------------------------
void llamaQuantizeRmsAtt(TASK_ARGS) { // 量化归一化后的结果并存储到TB_UNIT_XB_QUANTIZED中
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
}

void llamaSyncRmsAtt(TASK_ARGS) { // 同步量化归一化之后的数据,也就是TB_UNIT_XB_QUANTIZED中的数据,ROOT Node进行发送,Worker Node进行接收
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
}

void llamaQkv(TASK_ARGS) {
    TASK_VARIABLES;
    assert(block->kvCacheSlice->kvDim0 == block->k0Slice->d0);
    assert(block->kvCacheSlice->kvDim0 == block->v0Slice->d0);
    int memoryBudget = spec->memoryBudgetArray[block->sliceIndex];
    float *xbq = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED); // 获取量化归一化之后的TB_UNIT_XB_QUANTIZED中的数据
    float *k0 = &block->keyCache[transformer->pos * block->kvCacheSlice->kvDim0 * memoryBudget];
    float* v0 = &block->valueCache[transformer->pos * block->kvCacheSlice->kvDim0 * memoryBudget];
    block->q0mm->forward(xbq, block->qo0, nThreads, threadIndex, 0); // 计算xbq和q|k|v相乘的结果
    block->k0mm->forward(xbq, k0, nThreads, threadIndex, 0);
    block->v0mm->forward(xbq, v0, nThreads, threadIndex, 0);
}

void llamaRope(TASK_ARGS) { // 应用旋转位置编码到Q|K
    TASK_VARIABLES;
    int memoryBudget = spec->memoryBudgetArray[block->sliceIndex];
    float* k0 = &block->keyCache[transformer->pos * block->kvCacheSlice->kvDim0 * memoryBudget];
    transformer->rope->forward(true, block->qo0, transformer->pos, nThreads, threadIndex, memoryBudget);
    transformer->rope->forward(false, k0, transformer->pos, nThreads, threadIndex, memoryBudget);
}

void llamaMultiheadAtt(TASK_ARGS) { // 计算MHA的输出,将结果存储到TB_UNIT_XB中
    TASK_VARIABLES;
    SPLIT_RANGE_TO_THREADS(h0Start, h0End, 0, block->multiHeadAttSlice->nHeads0, nThreads, threadIndex); // 将MHA在线程中进行分配
    int sumMemoryBudget = 0;
    for (int value:spec->memoryBudgetArray) {
        sumMemoryBudget += value;
    }
    int midSumMemoryBudget = 0;
    for (int i = 0;i < transformer->sliceIndex;i++){
        midSumMemoryBudget += spec->memoryBudgetArray[i];
    }
    int memoryBudget = spec->memoryBudgetArray[transformer->sliceIndex];

    float* xb = (float*)transformer->buffer->getSliced(TB_UNIT_XB, transformer->sliceIndex, midSumMemoryBudget, sumMemoryBudget, true); // 从buffer中获取当前切片的缓冲数据
    int kvMul = spec->nHeads / spec->nKvHeads; // MHA中多个header共享一个Kv
    for (int h0 = h0Start; h0 < h0End; h0++) {
        // get the query vector for this head
        float* _q = block->qo0 + h0 * spec->headSize;
        float* _att = block->att + h0 * spec->seqLen;
        for (int t = 0; t <= transformer->pos; t++) {
            float* k = block->keyCache + t * block->kvCacheSlice->kvDim0 * memoryBudget + (h0 / kvMul) * spec->headSize;
            float score = dotProduct(_q, k, spec->headSize) / sqrtf(spec->headSize);
            _att[t] = score;
        }
        softmax(_att, transformer->pos + 1);
        float* hxb = xb + h0 * spec->headSize; // hxb是指向xb的后面n个head的指针,直接修改hxb本质上就是修改xb; 由于xb又是TB_UNIT_XB的指针,因此本质上是把计算的hxb累计到TB_UNIT_XB中
        memset(hxb, 0, spec->headSize * sizeof(float));
        for (int t = 0; t <= transformer->pos; t++) {
            float* _v = block->valueCache + t * block->kvCacheSlice->kvDim0 * memoryBudget + (h0 / kvMul) * spec->headSize;
            float a = _att[t];
            
            for (int i = 0; i < spec->headSize; i++) {
                hxb[i] += a * _v[i]; // 权重a应用到_v上,也就是将注意力得分应用到Value上
            }
        }
    }
}

void llamaQuantizeMultiheadAtt(TASK_ARGS) { // 对MHA的输出进行量化,量化结果存储在TB_UNIT_XB_QUANTIZED中
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
};

void llamaAtt(TASK_ARGS) { // 计算注意力的输出                                                
    TASK_VARIABLES;
    int sumMemoryBudget = 0;
    for (int value:spec->memoryBudgetArray) {
        sumMemoryBudget += value;
    }
    int midSumMemoryBudget = 0;
    for (int i = 0;i < transformer->sliceIndex;i++){
        midSumMemoryBudget += spec->memoryBudgetArray[i];
    }
    int memoryBudget = spec->memoryBudgetArray[transformer->sliceIndex];
    float* xbq0 = (float*)transformer->buffer->getSliced(TB_UNIT_XB_QUANTIZED, transformer->sliceIndex, midSumMemoryBudget, sumMemoryBudget, true);
    float* xbv0 = (float*)transformer->buffer->getSliced(TB_SLICED_XBV, transformer->sliceIndex, midSumMemoryBudget, sumMemoryBudget, true);

    block->wo0mm->forward(xbq0, xbv0, nThreads, threadIndex, 0);
}

void llamaQuantizeAtt(TASK_ARGS) { // 从TB_SLICED_XBV取出注意力的输出,量化,保存到TB_SLICED_XBV_QUANTIZED中
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XBV, TB_SLICED_XBV_QUANTIZED);
}
void llamaSyncAtt(TASK_ARGS) { // 同步注意力的计算结果
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XBV_QUANTIZED);
}

void llamaDequantizeAtt(TASK_ARGS) { // 对量化后的TB_SLICED_XBV_QUANTIZED注意力输出结果解量化, 保存到TB_SLICED_XBV中
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XBV_QUANTIZED, TB_SLICED_XBV);    
}

void llamaMergeAtt(TASK_ARGS) { // 合并所有切片的注意力输出
    TASK_VARIABLES;
    for (slice_index_t sliceIndex = 0; sliceIndex < spec->nSlices; sliceIndex++) {
        int sumMemoryBudget = 0;
        for (int value:spec->memoryBudgetArray) {
            sumMemoryBudget += value;
        }
        int midSumMemoryBudget = 0;
        for (int i = 0;i < sliceIndex;i++){
            midSumMemoryBudget += spec->memoryBudgetArray[i];
        }
        int memoryBudget = spec->memoryBudgetArray[sliceIndex];
        float* xbv = (float*)transformer->buffer->getSliced(TB_SLICED_XBV, sliceIndex, midSumMemoryBudget, sumMemoryBudget, false);
        add(transformer->x, xbv, spec->dim, nThreads, threadIndex);
    }
}

void llamaRmfFfn(TASK_ARGS) { // 计算RMS
    TASK_VARIABLES;
    if (threadIndex == 0) {
        transformer->rms = rms(transformer->x, spec->dim);
    }
}

void llamaRmfFfnNorm(TASK_ARGS) { // 基于RMS计算归一化值,并存储在TB_UNIT_XB中
    TASK_VARIABLES;
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB);
    float* x = (float*)transformer->x;

    rmsnorm(xb, x, transformer->rms, block->rmsFfn, spec->dim, nThreads, threadIndex);
}

void llamaQuantizeRmfFfn(TASK_ARGS) { // 从TB_UNIT_XB中取出归一化值,量化,存储到TB_UNIT_XB_QUANTIZED中
    TASK_VARIABLES;
    quantizeUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB, TB_UNIT_XB_QUANTIZED);
}

void llamaSyncFfn(TASK_ARGS) { // 同步TB_UNIT_XB_QUANTIZED中的数据
    TASK_VARIABLES;
    syncUnitBuffer(nThreads, threadIndex, ctx, TB_UNIT_XB_QUANTIZED);
}

void llamaFfn0(TASK_ARGS) {// 执行两个线性层并执行激活函数,结果保存在hb0中,也就是TB_SLICED_HB中
    TASK_VARIABLES;
    int sumMemoryBudget = 0;
    for (int value:spec->memoryBudgetArray) {
        sumMemoryBudget += value;
    }
    int midSumMemoryBudget = 0;
    for (int i = 0;i < transformer->sliceIndex;i++){
        midSumMemoryBudget += spec->memoryBudgetArray[i];
    }
    int memoryBudget = spec->memoryBudgetArray[transformer->sliceIndex];
    float* xb = (float*)transformer->buffer->getUnit(TB_UNIT_XB_QUANTIZED);

    float* hb0 = (float*)transformer->buffer->getSliced(TB_SLICED_HB, transformer->sliceIndex, midSumMemoryBudget, sumMemoryBudget, true);
    
    block->w10mm->forward(xb, hb0, nThreads, threadIndex, 0); // xb为输入,执行前向传播,结果保存到hb0
    block->w30mm->forward(xb, block->hb20, nThreads, threadIndex, 0);
    if (spec->hiddenAct == SILU) {
        silu(hb0, block->w10Slice->d0 * memoryBudget, nThreads, threadIndex);
    } else if (spec->hiddenDim == GELU) {
        gelu(hb0, block->w10Slice->d0 * memoryBudget, nThreads, threadIndex);
    } else {
        assert(false);
    }
    mul(hb0, block->hb20, block->w10Slice->d0 * memoryBudget, nThreads, threadIndex);
}

void llamaFfn1(TASK_ARGS) { // 从TB_SLICED_HB取出前向传播的结果,量化,保存到TB_SLICED_HB_QUANTIZED中
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, true, TB_SLICED_HB, TB_SLICED_HB_QUANTIZED);
    
}

void llamaFfn2(TASK_ARGS) { // 从TB_SLICED_HB_QUANTIZED取出input,进行第三层前向传播,结果保存在TB_SLICED_XBV中
    TASK_VARIABLES;
    int sumMemoryBudget = 0;
    for (int value:spec->memoryBudgetArray) {
        sumMemoryBudget += value;
    }
    int midSumMemoryBudget = 0;
    for (int i = 0;i < transformer->sliceIndex;i++){
        midSumMemoryBudget += spec->memoryBudgetArray[i];
    }
    int memoryBudget = spec->memoryBudgetArray[transformer->sliceIndex];
    float *hb = (float*)transformer->buffer->getSliced(TB_SLICED_HB_QUANTIZED, transformer->sliceIndex, midSumMemoryBudget, sumMemoryBudget, true);
    float *xbv = (float*)transformer->buffer->getSliced(TB_SLICED_XBV, transformer->sliceIndex, midSumMemoryBudget, sumMemoryBudget, true);
    block->w20mm->forward(hb, xbv, nThreads, threadIndex, 0);
}

void llamaQuantizeFfn2(TASK_ARGS) {// TB_SLICED_XBV --> TB_SLICED_XBV_QUANTIZED
    TASK_VARIABLES;
    quantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XBV, TB_SLICED_XBV_QUANTIZED);
}

void llamaSyncFfn2(TASK_ARGS) { // 同步第三个线性层的量化输出
    TASK_VARIABLES;
    syncSliceOfSlicedBuffer(nThreads, threadIndex, ctx, TB_SLICED_XBV_QUANTIZED);
}

void llamaDequantizeFfn2(TASK_ARGS) { // 解第三个线性层的量化输出
    TASK_VARIABLES;
    dequantizeSlicedBuffer(nThreads, threadIndex, ctx, false, TB_SLICED_XBV_QUANTIZED, TB_SLICED_XBV);
}

void llamaMergeFfn2(TASK_ARGS) { // 合并所有slice的前向传播结果,保存到transformer->x中
    TASK_VARIABLES;
    for (slice_index_t sliceIndex = 0; sliceIndex < spec->nSlices; sliceIndex++) {
        int sumMemoryBudget = 0;
        for (int value:spec->memoryBudgetArray) {
            sumMemoryBudget += value;
        }
        int midSumMemoryBudget = 0;
        for (int i = 0;i < sliceIndex;i++){
            midSumMemoryBudget += spec->memoryBudgetArray[i];
        }
        int memoryBudget = spec->memoryBudgetArray[sliceIndex];
        float* xbv = (float*)transformer->buffer->getSliced(TB_SLICED_XBV, sliceIndex, midSumMemoryBudget, sumMemoryBudget, false);
        add(transformer->x, xbv, spec->dim, nThreads, threadIndex);
    }
}

void llamaNextBlock(TASK_ARGS) { // 移动到下一个block
    TASK_VARIABLES;
    if (threadIndex == 0) {
        ctx->currentBlockIndex++;
    }
}

void llamaRmsFinal(TASK_ARGS) { // 对最终的输出进行rms
    TASK_VARIABLES;
    if (threadIndex == 0) {
        float* x = transformer->x;
        transformer->rms = rms(x, spec->dim);
    }
}

void llamaRmsFinalNorm(TASK_ARGS) { // 基于最终输出的rms也就是transformer->rms计算归一化结果
    TASK_VARIABLES;
    float* x = transformer->x;
    rmsnorm(x, x, transformer->rms, (float*)transformer->rmsFinal, spec->dim, nThreads, threadIndex);
}

void llamaFinalize(TASK_ARGS) { // 执行最终的线性层,将结果存储在transformer->logits中
    TASK_VARIABLES;
    transformer->wclsMm->forward(transformer->x, transformer->logits, nThreads, threadIndex, 0);
}

TransformerArch buildLlamaArch(TransformerSpec* spec) { // return
    TransformerArch a;

    // inference

    a.I(sendPos, TASK_TYPE_TRANSFER); // ROOT Node把transformer->pos中的数据发送给Worker Node
    for (int i = 0; i < spec->nLayers; i++) {
        a.I(llamaRmsAtt, TASK_TYPE_INFERENCE); // 仅在线程0中执行,计算Transformer的block的RMS值
        a.I(llamaRmsAttNorm, TASK_TYPE_INFERENCE); // 通过上面计算的rms快速计算归一化值并存储在TB_UNIT_XB中
        a.I(llamaQuantizeRmsAtt, TASK_TYPE_INFERENCE);// 量化归一化后的结果并存储到TB_UNIT_XB_QUANTIZED中
        a.I(llamaSyncRmsAtt, TASK_TYPE_TRANSFER);// 同步量化归一化之后的数据,也就是TB_UNIT_XB_QUANTIZED中的数据,ROOT Node进行发送,Worker Node进行接收
        a.I(llamaQkv, TASK_TYPE_INFERENCE);// 将上面的量化之后的归一化结果取出,计算q|k|v
        a.I(llamaRope, TASK_TYPE_INFERENCE);// 应用旋转位置编码到Q|K
        a.I(llamaMultiheadAtt, TASK_TYPE_INFERENCE);
        /*
        计算MHA的输出,存储到TB_UNIT_XB中
            1. 计算查询和键的点积，得到注意力得分。
            2. 对得分进行softmax变换，得到注意力权重。
            3. 使用注意力权重对值进行加权求和，得到输出向量。
        */
        a.I(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);// 对MHA的输出进行量化,量化结果存储在TB_UNIT_XB_QUANTIZED中
        a.I(llamaAtt, TASK_TYPE_INFERENCE);// 计算注意力的输出
        a.I(llamaQuantizeAtt, TASK_TYPE_INFERENCE);// 从TB_SLICED_XBV取出注意力的输出,量化,保存到TB_SLICED_XBV_QUANTIZED中
        a.I(llamaSyncAtt, TASK_TYPE_TRANSFER);// 同步注意力的计算结果
        a.I(llamaDequantizeAtt, TASK_TYPE_INFERENCE);// 对量化后的TB_SLICED_XBV_QUANTIZED注意力输出结果解量化, 保存到TB_SLICED_XBV中
        a.I(llamaMergeAtt, TASK_TYPE_INFERENCE);// 合并所有切片的注意力输出
        a.I(llamaRmfFfn, TASK_TYPE_INFERENCE);// 计算RMS
        a.I(llamaRmfFfnNorm, TASK_TYPE_INFERENCE);// 基于RMS计算Norm
        a.I(llamaQuantizeRmfFfn, TASK_TYPE_INFERENCE);// 对RMS计算出的归一化结果进行量化
        a.I(llamaSyncFfn, TASK_TYPE_TRANSFER);// 同步量化结果
        /*
        注意:llamaSyncFfn是ROOT Node将自己的归一化计算结果同步给ROOT Node,因此对于ROOT Node是writeMany,对于Worker Node是readMany
        */
        a.I(llamaFfn0, TASK_TYPE_INFERENCE);// 执行两个线性层并执行激活函数,结果保存在hb0中,也就是TB_SLICED_HB中
        a.I(llamaFfn1, TASK_TYPE_INFERENCE);// 从TB_SLICED_HB取出前向传播的结果,量化,保存到TB_SLICED_HB_QUANTIZED中
        a.I(llamaFfn2, TASK_TYPE_INFERENCE);// 从TB_SLICED_HB_QUANTIZED取出input,进行第三层前向传播,结果保存在TB_SLICED_XBV中
        a.I(llamaQuantizeFfn2, TASK_TYPE_INFERENCE);// 量化
        a.I(llamaSyncFfn2, TASK_TYPE_TRANSFER);// 同步
        /*
        注意:llamaSyncFfn2是Worker Node将自己的前向传播和激活函数计算结果同步给ROOT Node,因此对于ROOT Node是readMany,对于Worker Node是writeMany
        */
        a.I(llamaDequantizeFfn2, TASK_TYPE_INFERENCE);// 解量化0x1020d3bc4
        a.I(llamaMergeFfn2, TASK_TYPE_INFERENCE);// 合并所有slice的前向传播结果
        a.I(llamaNextBlock, TASK_TYPE_INFERENCE);// 移动到下一个block
    }
    a.I(llamaRmsFinal, TASK_TYPE_INFERENCE);// 对最终的输出进行rms
    a.I(llamaRmsFinalNorm, TASK_TYPE_INFERENCE);// 基于最终输出的rms也就是transformer->rms计算归一化结果
    a.I(llamaFinalize, TASK_TYPE_INFERENCE);// 执行最终的线性层,将结果存储在transformer->logits中

    // worker

    for (int i = 0; i < spec->nLayers; i++) {
        a.W(llamaSyncRmsAtt, TASK_TYPE_TRANSFER); // 接收归一化之后的结果
        a.W(llamaQkv, TASK_TYPE_INFERENCE);// 从接收的结果中计算Qkv
        a.W(llamaRope, TASK_TYPE_INFERENCE);// 旋转编码
        a.W(llamaMultiheadAtt, TASK_TYPE_INFERENCE);// 计算MHA的输出
        a.W(llamaQuantizeMultiheadAtt, TASK_TYPE_INFERENCE);// 对MHA的输出进行量化,量化结果存储在TB_UNIT_XB_QUANTIZED中
        a.W(llamaAtt, TASK_TYPE_INFERENCE);// 计算注意力的输出
        a.W(llamaQuantizeAtt, TASK_TYPE_INFERENCE);// 量化注意力的输出
        a.W(llamaSyncAtt, TASK_TYPE_TRANSFER); // 同步
        a.W(llamaSyncFfn, TASK_TYPE_TRANSFER);// 同步激活层量化记过
        a.W(llamaFfn0, TASK_TYPE_INFERENCE);
        a.W(llamaFfn1, TASK_TYPE_INFERENCE);
        a.W(llamaFfn2, TASK_TYPE_INFERENCE);// 执行三层前向传播和激活函数
        a.W(llamaQuantizeFfn2, TASK_TYPE_INFERENCE);// 量化
        a.W(llamaSyncFfn2, TASK_TYPE_TRANSFER);// 同步前向传播和激活的结果
        a.W(llamaNextBlock, TASK_TYPE_INFERENCE);
    }
    return a;
}
