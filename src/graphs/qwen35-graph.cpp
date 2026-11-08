#include "qwen35-graph.hpp"
#include "../nn/nn-config-builder.hpp"
#include "../mmap.hpp"
#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

static bool isFullAttentionLayer(const Qwen35LlmHeader *header, NnUint layerIndex) {
    return (layerIndex + 1u) % header->fullAttentionInterval == 0u;
}

class Qwen35LlmModel : public LlmModel {
public:
    Qwen35LlmModel() : LlmModel({LLM_MAGIC_QWEN35}) {}

    std::unique_ptr<LlmHeader> loadHeader(
        FILE *file, int magic, NnUint maxSeqLen, NnFloatType syncType) const override;
    void printHeader(const LlmHeader *header) const override;
    std::unique_ptr<LlmGraph> buildGraph(
        LlmHeader *header, NnUint nNodes, NnUint nBatches) const override;
    void loadGraph(const char *path, LlmGraph *graph, NnRootWeightLoader *loader) const override;
};

std::unique_ptr<LlmHeader> Qwen35LlmModel::loadHeader(
    FILE *file, int magic, NnUint maxSeqLen, NnFloatType syncType) const {
    std::unique_ptr<Qwen35LlmHeader> header(new Qwen35LlmHeader());
    header->magic = magic;
    header->weightType = F_UNK;
    header->ropeTheta = 10000000.0f;
    header->normEpsilon = 1e-6f;
    header->fullAttentionInterval = 4u;
    header->linearConvKernelDim = 4u;
    header->linearKeyHeadDim = 128u;
    header->linearValueHeadDim = 128u;
    header->linearNKeyHeads = 16u;
    header->linearNValueHeads = 32u;
    bool hasArchType = false;

    if (fread(&header->headerSize, sizeof(int), 1, file) != 1)
        throw std::runtime_error("Cannot read Qwen3.5 header size");
    if (header->headerSize < 2u * sizeof(int))
        throw std::runtime_error("Invalid Qwen3.5 header size");

    const NnSize dataSize = header->headerSize - 2u * sizeof(int);
    if (dataSize % (2u * sizeof(int)) != 0u)
        throw std::runtime_error("Invalid Qwen3.5 header key/value data");
    std::vector<int> buffer(dataSize / sizeof(int));
    if (dataSize > 0 && fread(buffer.data(), dataSize, 1, file) != 1)
        throw std::runtime_error("Cannot read Qwen3.5 header values");

    for (NnSize i = 0; i < buffer.size(); i += 2) {
        int key = buffer[i];
        int value = buffer[i + 1];
        if (key == QWEN35_VERSION) header->version = value;
        else if (key == QWEN35_ARCH_TYPE) {
            header->archType = (Qwen35LlmArchType)value;
            hasArchType = true;
        }
        else if (key == QWEN35_DIM) header->dim = value;
        else if (key == QWEN35_HIDDEN_DIM) header->hiddenDim = value;
        else if (key == QWEN35_N_LAYERS) header->nLayers = value;
        else if (key == QWEN35_N_HEADS) header->nHeads = value;
        else if (key == QWEN35_N_KV_HEADS) header->nKvHeads = value;
        else if (key == QWEN35_VOCAB_SIZE) header->vocabSize = value;
        else if (key == QWEN35_SEQ_LEN) header->seqLen = value;
        else if (key == QWEN35_ROPE_THETA) header->ropeTheta = (float)value;
        else if (key == QWEN35_WEIGHT_FLOAT_TYPE) header->weightType = (NnFloatType)value;
        else if (key == QWEN35_HEAD_DIM) header->headDim = value;
        else if (key == QWEN35_ROTARY_DIM) header->rotaryDim = value;
        else if (key == QWEN35_NORM_EPSILON) header->normEpsilon = convertNormEpsilon(value);
        else if (key == QWEN35_FULL_ATTENTION_INTERVAL) header->fullAttentionInterval = value;
        else if (key == QWEN35_LINEAR_CONV_KERNEL_DIM) header->linearConvKernelDim = value;
        else if (key == QWEN35_LINEAR_KEY_HEAD_DIM) header->linearKeyHeadDim = value;
        else if (key == QWEN35_LINEAR_VALUE_HEAD_DIM) header->linearValueHeadDim = value;
        else if (key == QWEN35_LINEAR_N_KEY_HEADS) header->linearNKeyHeads = value;
        else if (key == QWEN35_LINEAR_N_VALUE_HEADS) header->linearNValueHeads = value;
        else throw std::runtime_error("Unsupported Qwen3.5 header key");
    }

    if (!hasArchType)
        throw std::runtime_error("Qwen3.5 model does not specify architecture type");
    if (header->archType != QWEN35)
        throw std::runtime_error("Unsupported Qwen3.5 architecture");
    if (header->weightType == F_UNK)
        throw std::runtime_error("Qwen3.5 model does not specify weight type");
    if (header->headDim == 0u)
        header->headDim = header->dim / header->nHeads;
    if (header->rotaryDim == 0u)
        header->rotaryDim = header->headDim / 4u;
    if (header->fullAttentionInterval == 0u)
        throw std::runtime_error("Qwen3.5 full attention interval must be non-zero");

    header->qDim = header->headDim * header->nHeads;
    header->kvDim = header->headDim * header->nKvHeads;
    header->origSeqLen = header->seqLen;
    if (maxSeqLen > 0 && header->seqLen > maxSeqLen)
        header->seqLen = maxSeqLen;
    header->syncType = syncType;
    return std::unique_ptr<LlmHeader>(header.release());
}

void Qwen35LlmModel::printHeader(const LlmHeader *baseHeader) const {
    const Qwen35LlmHeader *header = static_cast<const Qwen35LlmHeader *>(baseHeader);
    printf("💡 Format: Qwen3.5\n");
    printf("💡 Arch: Qwen3.5\n");
    printf("💡 Dim: %u\n", header->dim);
    printf("💡 HiddenDim: %u\n", header->hiddenDim);
    printf("💡 HeadDim: %u\n", header->headDim);
    printf("💡 RotaryDim: %u\n", header->rotaryDim);
    printf("💡 VocabSize: %u\n", header->vocabSize);
    printf("💡 nLayers: %u\n", header->nLayers);
    printf("💡 nHeads: %u\n", header->nHeads);
    printf("💡 nKvHeads: %u\n", header->nKvHeads);
    printf("💡 FullAttentionInterval: %u\n", header->fullAttentionInterval);
    printf("💡 LinearHeads: k=%u, v=%u\n", header->linearNKeyHeads, header->linearNValueHeads);
    printf("💡 LinearHeadDim: k=%u, v=%u\n", header->linearKeyHeadDim, header->linearValueHeadDim);
    printf("💡 LinearConvKernelDim: %u\n", header->linearConvKernelDim);
    if (header->seqLen != header->origSeqLen)
        printf("💡 OrigSeqLen: %u\n", header->origSeqLen);
    printf("💡 SeqLen: %u\n", header->seqLen);
    printf("💡 NormEpsilon: %f\n", header->normEpsilon);
    printf("💡 RopeTheta: %.0f\n", header->ropeTheta);
}

std::unique_ptr<LlmGraph> Qwen35LlmModel::buildGraph(
    LlmHeader *baseHeader, NnUint nNodes, NnUint nBatches) const {
    Qwen35LlmHeader *h = static_cast<Qwen35LlmHeader *>(baseHeader);
    const NnUint linearKeyDim = h->linearNKeyHeads * h->linearKeyHeadDim;
    const NnUint linearValueDim = h->linearNValueHeads * h->linearValueHeadDim;
    const NnUint localLinearKeyHeads = h->linearNKeyHeads / nNodes;
    const NnUint localLinearValueHeads = h->linearNValueHeads / nNodes;

    assert(h->linearNKeyHeads % nNodes == 0);
    assert(h->linearNValueHeads % nNodes == 0);

    std::unique_ptr<Qwen35Graph> graph(new Qwen35Graph());
    Qwen35Graph &n = *graph;
    n.header = h;
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);
    n.qkRmsNormSize = size1D(F_32, h->headDim);
    n.qSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->qDim);
    n.kSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.vSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.woSlice = sliceColMatmul(h->weightType, nNodes, h->qDim, h->dim);
    n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->hiddenDim);
    n.w2Slice = sliceColMatmul(h->weightType, nNodes, h->hiddenDim, h->dim);
    n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->hiddenDim);
    n.wclsSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->vocabSize);

    n.gateSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->qDim);
    n.linearQSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, linearKeyDim);
    n.linearKSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, linearKeyDim);
    n.linearVSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, linearValueDim);
    n.linearZSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, linearValueDim);
    n.linearASlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->linearNValueHeads);
    n.linearBSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->linearNValueHeads);
    n.linearOutSlice = sliceColMatmul(h->weightType, nNodes, linearValueDim, h->dim);
    const NnRowMatmulSlice &gateSlice = n.gateSlice;
    const NnRowMatmulSlice &linearQSlice = n.linearQSlice;
    const NnRowMatmulSlice &linearKSlice = n.linearKSlice;
    const NnRowMatmulSlice &linearVSlice = n.linearVSlice;
    const NnRowMatmulSlice &linearZSlice = n.linearZSlice;
    const NnRowMatmulSlice &linearASlice = n.linearASlice;
    const NnRowMatmulSlice &linearBSlice = n.linearBSlice;
    const NnColMatmulSlice &linearOutSlice = n.linearOutSlice;
    const NnKvCacheSlice kvCacheSlice = sliceKvCache(h->kvDim, h->seqLen, nNodes);
    const NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(h->nHeads, h->seqLen, nNodes, nBatches);

    NnNetConfigBuilder netBuilder(nNodes, nBatches);
    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.outputPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    const NnUint zqPipeIndex = netBuilder.addPipe("ZQ", size2D(h->syncType, nBatches, h->dim * nNodes));
    netBuilder.addPreSync(n.positionPipeIndex);
    n.netConfig = netBuilder.build();
    n.nodeConfigs = new NnNodeConfig[nNodes]();

    for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        NnNodeConfigBuilder nodeBuilder(nodeIndex);
        const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
        const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
        const NnUint yqBufferIndex = h->syncType == F_32
            ? yBufferIndex
            : nodeBuilder.addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));
        const NnUint invRmsBufferIndex = nodeBuilder.addBuffer(
            "inv_rms", size2D(F_32, nBatches, std::max(h->nHeads / nNodes, localLinearValueHeads)));
        const NnUint outputSliceBufferIndex = nodeBuilder.addBuffer("output", size2D(F_32, nBatches, h->dim));
        const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer(
            "lg", size2D(F_32, nBatches, h->vocabSize / nNodes));
        const NnUint unusedExpertIndexesBufferIndex = nodeBuilder.addBuffer("unused_expert", size1D(F_32, 1));
        const NnUint ffGateBufferIndex = nodeBuilder.addBuffer(
            "ff_gate", size2D(F_32, nBatches, n.w1Slice.d0));
        const NnUint ffGateQBufferIndex = h->syncType == F_32
            ? ffGateBufferIndex
            : nodeBuilder.addBuffer("q_ff_gate", size2D(h->syncType, nBatches, n.w1Slice.d0));
        const NnUint ffUpBufferIndex = nodeBuilder.addBuffer(
            "ff_up", size2D(F_32, nBatches, n.w3Slice.d0));

        NnSegmentConfigBuilder start;
        if (nodeIndex == 0) {
            start.addOp(OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, n.tokenPipeIndex),
                pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                n.tokenEmbeddingSize, NnEmbeddingOpConfig{});
        }
        start.addSync(n.xPipeIndex, SYNC_WITH_ROOT);
        nodeBuilder.addSegment(start.build());

        for (NnUint layerIndex = 0; layerIndex < h->nLayers; layerIndex++) {
            NnSegmentConfigBuilder mixer;
            if (layerIndex == 0) {
                mixer.addOp(OP_CAST, "block_cast_x", layerIndex,
                    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(), NnCastOpCodeConfig{});
            } else {
                mixer.addOp(OP_MERGE_ADD, "block_merge_add", layerIndex,
                    pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(), NnMergeAddOpCodeConfig{});
            }
            mixer.addOp(OP_INV_RMS, "block_norm_pre_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(), NnInvRmsOpConfig{h->normEpsilon, 1});
            mixer.addOp(OP_RMS_NORM_ZERO_CENTERED, "block_norm_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize, NnRmsNormOpConfig{invRmsBufferIndex, 1});
            if (yBufferIndex != yqBufferIndex) {
                mixer.addOp(OP_CAST, "block_cast_y", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    size0(), NnCastOpCodeConfig{});
            }

            if (isFullAttentionLayer(h, layerIndex)) {
                const NnRopeSlice ropeSlice = sliceRope(
                    ROPE_FALCON, h->qDim, h->kvDim, h->nKvHeads, nNodes,
                    h->seqLen, h->headDim, h->ropeTheta, nodeIndex);
                const NnUint qBufferIndex = nodeBuilder.addBuffer("full_q", size2D(F_32, nBatches, n.qSlice.d0));
                const NnUint kTempBufferIndex = nodeBuilder.addBuffer("full_k_temp", size2D(F_32, nBatches, n.kSlice.d0));
                const NnUint vTempBufferIndex = nodeBuilder.addBuffer("full_v_temp", size2D(F_32, nBatches, n.vSlice.d0));
                const NnUint gateBufferIndex = nodeBuilder.addBuffer("full_gate", size2D(F_32, nBatches, gateSlice.d0));
                const NnUint attentionOutputBufferIndex = nodeBuilder.addBuffer("full_output", size2D(F_32, nBatches, n.qSlice.d0));
                const NnUint attentionOutputQBufferIndex = h->syncType == F_32
                    ? attentionOutputBufferIndex
                    : nodeBuilder.addBuffer("q_full_output", size2D(h->syncType, nBatches, n.qSlice.d0));
                const NnUint keyCacheBufferIndex = nodeBuilder.addBuffer("full_k_cache", kvCacheSlice.keySize);
                const NnUint valueCacheBufferIndex = nodeBuilder.addBuffer("full_v_cache", kvCacheSlice.valueSize);
                const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("partial_rope_cache", ropeSlice.cacheSize);
                const NnUint attentionBufferIndex = nodeBuilder.addBuffer("full_att", multiHeadAttSlice.attSize);

                mixer.addOp(OP_MATMUL, "block_matmul_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size2D(h->weightType, n.qSlice.n, n.qSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_matmul_gate", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, gateBufferIndex),
                    size2D(h->weightType, gateSlice.n, gateSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_matmul_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    size2D(h->weightType, n.kSlice.n, n.kSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_matmul_v", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                    size2D(h->weightType, n.vSlice.n, n.vSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});

                const NnUint localQHeads = h->nHeads / nNodes;
                const NnUint localKvHeads = h->nKvHeads / nNodes;
                mixer.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(), NnInvRmsOpConfig{h->normEpsilon, localQHeads});
                mixer.addOp(OP_RMS_NORM_ZERO_CENTERED, "block_norm_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size1D(F_32, h->headDim), NnRmsNormOpConfig{invRmsBufferIndex, localQHeads});
                mixer.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(), NnInvRmsOpConfig{h->normEpsilon, localKvHeads});
                mixer.addOp(OP_RMS_NORM_ZERO_CENTERED, "block_norm_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    size1D(F_32, h->headDim), NnRmsNormOpConfig{invRmsBufferIndex, localKvHeads});
                mixer.addOp(OP_PARTIAL_ROPE, "block_partial_rope_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size0(), NnPartialRopeOpConfig{1, n.positionPipeIndex, ropeCacheBufferIndex, h->rotaryDim, ropeSlice});
                mixer.addOp(OP_PARTIAL_ROPE, "block_partial_rope_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    size0(), NnPartialRopeOpConfig{0, n.positionPipeIndex, ropeCacheBufferIndex, h->rotaryDim, ropeSlice});
                mixer.addOp(OP_SHIFT, "block_shift_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerRawConfig(SRC_BUFFER, keyCacheBufferIndex),
                    size0(), NnShiftOpCodeConfig{n.positionPipeIndex});
                mixer.addOp(OP_SHIFT, "block_shift_v", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, vTempBufferIndex), pointerRawConfig(SRC_BUFFER, valueCacheBufferIndex),
                    size0(), NnShiftOpCodeConfig{n.positionPipeIndex});
                mixer.addOp(OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, attentionOutputBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, attentionOutputBufferIndex), size0(),
                    NnMultiHeadAttOpConfig{multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
                        h->nKvHeads, h->headDim, h->seqLen, n.qSlice.d0, kvCacheSlice.kvDim0,
                        n.positionPipeIndex, qBufferIndex, keyCacheBufferIndex, valueCacheBufferIndex,
                        attentionBufferIndex});
                mixer.addOp(OP_SIGMOID_MUL, "block_attention_gate", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, attentionOutputBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, attentionOutputBufferIndex), size0(),
                    NnSigmoidMulOpConfig{gateBufferIndex});
                if (attentionOutputBufferIndex != attentionOutputQBufferIndex) {
                    mixer.addOp(OP_CAST, "block_cast_attention", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, attentionOutputBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, attentionOutputQBufferIndex), size0(), NnCastOpCodeConfig{});
                }
                mixer.addOp(OP_MATMUL, "block_matmul_wo", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, attentionOutputQBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, outputSliceBufferIndex),
                    size2D(h->weightType, n.woSlice.n0, n.woSlice.d),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
            } else {
                const NnUint qBufferIndex = nodeBuilder.addBuffer("linear_q", size2D(F_32, nBatches, linearQSlice.d0));
                const NnUint kBufferIndex = nodeBuilder.addBuffer("linear_k", size2D(F_32, nBatches, linearKSlice.d0));
                const NnUint vBufferIndex = nodeBuilder.addBuffer("linear_v", size2D(F_32, nBatches, linearVSlice.d0));
                const NnUint zBufferIndex = nodeBuilder.addBuffer("linear_z", size2D(F_32, nBatches, linearZSlice.d0));
                const NnUint aBufferIndex = nodeBuilder.addBuffer("linear_a", size2D(F_32, nBatches, linearASlice.d0));
                const NnUint bBufferIndex = nodeBuilder.addBuffer("linear_b", size2D(F_32, nBatches, linearBSlice.d0));
                const NnUint qConvStateBufferIndex = nodeBuilder.addBuffer(
                    "linear_q_conv_state", size2D(F_32, linearQSlice.d0, h->linearConvKernelDim), true);
                const NnUint kConvStateBufferIndex = nodeBuilder.addBuffer(
                    "linear_k_conv_state", size2D(F_32, linearKSlice.d0, h->linearConvKernelDim), true);
                const NnUint vConvStateBufferIndex = nodeBuilder.addBuffer(
                    "linear_v_conv_state", size2D(F_32, linearVSlice.d0, h->linearConvKernelDim), true);
                const NnUint recurrentStateBufferIndex = nodeBuilder.addBuffer(
                    "linear_recurrent_state", size3D(F_32, localLinearValueHeads,
                        h->linearKeyHeadDim, h->linearValueHeadDim), true);
                const NnUint coreOutputBufferIndex = nodeBuilder.addBuffer(
                    "linear_core_output", size2D(F_32, nBatches, linearVSlice.d0));
                const NnUint coreOutputQBufferIndex = h->syncType == F_32
                    ? coreOutputBufferIndex
                    : nodeBuilder.addBuffer("q_linear_core_output", size2D(h->syncType, nBatches, linearVSlice.d0));

                mixer.addOp(OP_MATMUL, "block_linear_matmul_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size2D(h->weightType, linearQSlice.n, linearQSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_linear_matmul_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, kBufferIndex),
                    size2D(h->weightType, linearKSlice.n, linearKSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_linear_matmul_v", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, vBufferIndex),
                    size2D(h->weightType, linearVSlice.n, linearVSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_linear_matmul_z", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, zBufferIndex),
                    size2D(h->weightType, linearZSlice.n, linearZSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_linear_matmul_a", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, aBufferIndex),
                    size2D(h->weightType, linearASlice.n, linearASlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_MATMUL, "block_linear_matmul_b", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, bBufferIndex),
                    size2D(h->weightType, linearBSlice.n, linearBSlice.d0),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
                mixer.addOp(OP_CAUSAL_CONV1D, "block_linear_conv_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size2D(F_32, linearQSlice.d0, h->linearConvKernelDim),
                    NnCausalConv1DOpConfig{qConvStateBufferIndex, n.positionPipeIndex, h->linearConvKernelDim, 1});
                mixer.addOp(OP_CAUSAL_CONV1D, "block_linear_conv_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kBufferIndex), pointerBatchConfig(SRC_BUFFER, kBufferIndex),
                    size2D(F_32, linearKSlice.d0, h->linearConvKernelDim),
                    NnCausalConv1DOpConfig{kConvStateBufferIndex, n.positionPipeIndex, h->linearConvKernelDim, 1});
                mixer.addOp(OP_CAUSAL_CONV1D, "block_linear_conv_v", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, vBufferIndex), pointerBatchConfig(SRC_BUFFER, vBufferIndex),
                    size2D(F_32, linearVSlice.d0, h->linearConvKernelDim),
                    NnCausalConv1DOpConfig{vConvStateBufferIndex, n.positionPipeIndex, h->linearConvKernelDim, 1});
                mixer.addOp(OP_GATED_DELTA_NET, "block_linear_delta", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, coreOutputBufferIndex),
                    size2D(F_32, 2, localLinearValueHeads),
                    NnGatedDeltaNetOpConfig{kBufferIndex, vBufferIndex, aBufferIndex, bBufferIndex,
                        recurrentStateBufferIndex, n.positionPipeIndex, localLinearKeyHeads,
                        localLinearValueHeads, h->linearKeyHeadDim, h->linearValueHeadDim});
                mixer.addOp(OP_INV_RMS, "block_linear_norm_pre", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, coreOutputBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex), size0(),
                    NnInvRmsOpConfig{h->normEpsilon, localLinearValueHeads});
                mixer.addOp(OP_RMS_NORM, "block_linear_norm", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, coreOutputBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, coreOutputBufferIndex),
                    size1D(F_32, h->linearValueHeadDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, localLinearValueHeads});
                mixer.addOp(OP_SILU, "block_linear_z_act", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, zBufferIndex), pointerBatchConfig(SRC_BUFFER, zBufferIndex),
                    size0(), NnSiluOpCodeConfig{});
                mixer.addOp(OP_MUL, "block_linear_z_gate", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, coreOutputBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, coreOutputBufferIndex), size0(),
                    NnMulOpCodeConfig{zBufferIndex});
                if (coreOutputBufferIndex != coreOutputQBufferIndex) {
                    mixer.addOp(OP_CAST, "block_cast_linear_output", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, coreOutputBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, coreOutputQBufferIndex), size0(), NnCastOpCodeConfig{});
                }
                mixer.addOp(OP_MATMUL, "block_matmul_wo", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, coreOutputQBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, outputSliceBufferIndex),
                    size2D(h->weightType, linearOutSlice.n0, linearOutSlice.d),
                    NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
            }

            mixer.addOp(OP_CAST, "block_cast_mixer_output", layerIndex,
                pointerBatchConfig(SRC_BUFFER, outputSliceBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex), size0(), NnCastOpCodeConfig{});
            mixer.addSync(zqPipeIndex, SYNC_NODE_SLICES);
            nodeBuilder.addSegment(mixer.build());

            NnSegmentConfigBuilder ff;
            ff.addOp(OP_MERGE_ADD, "block_merge_add2", layerIndex,
                pointerBatchConfig(SRC_PIPE, zqPipeIndex), pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                size0(), NnMergeAddOpCodeConfig{});
            ff.addOp(OP_INV_RMS, "block_norm_pre_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(), NnInvRmsOpConfig{h->normEpsilon, 1});
            ff.addOp(OP_RMS_NORM_ZERO_CENTERED, "block_norm_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize, NnRmsNormOpConfig{invRmsBufferIndex, 1});
            if (yBufferIndex != yqBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_y3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    size0(), NnCastOpCodeConfig{});
            }
            ff.addOp(OP_MATMUL, "block_matmul_w1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, ffGateBufferIndex),
                size2D(h->weightType, n.w1Slice.n, n.w1Slice.d0),
                NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
            ff.addOp(OP_MATMUL, "block_matmul_w3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, ffUpBufferIndex),
                size2D(h->weightType, n.w3Slice.n, n.w3Slice.d0),
                NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
            ff.addOp(OP_SILU, "block_act", layerIndex,
                pointerBatchConfig(SRC_BUFFER, ffGateBufferIndex), pointerBatchConfig(SRC_BUFFER, ffGateBufferIndex),
                size0(), NnSiluOpCodeConfig{});
            ff.addOp(OP_MUL, "block_mul", layerIndex,
                pointerBatchConfig(SRC_BUFFER, ffGateBufferIndex), pointerBatchConfig(SRC_BUFFER, ffGateBufferIndex),
                size0(), NnMulOpCodeConfig{ffUpBufferIndex});
            if (ffGateBufferIndex != ffGateQBufferIndex) {
                ff.addOp(OP_CAST, "block_cast_d2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, ffGateBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, ffGateQBufferIndex), size0(), NnCastOpCodeConfig{});
            }
            ff.addOp(OP_MATMUL, "block_matmul_w2", layerIndex,
                pointerBatchConfig(SRC_BUFFER, ffGateQBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                size2D(h->weightType, n.w2Slice.n0, n.w2Slice.d),
                NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
            ff.addOp(OP_CAST, "block_cast_d3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
                size0(), NnCastOpCodeConfig{});
            ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);
            nodeBuilder.addSegment(ff.build());
        }

        NnSegmentConfigBuilder end;
        end.addOp(OP_MERGE_ADD, "final_merge_add", 0,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex), pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            size0(), NnMergeAddOpCodeConfig{});
        end.addOp(OP_INV_RMS, "final_norm_pre", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
            size0(), NnInvRmsOpConfig{h->normEpsilon, 1});
        end.addOp(OP_RMS_NORM_ZERO_CENTERED, "final_norm", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex), pointerBatchConfig(SRC_BUFFER, yBufferIndex),
            n.rmsNormSize, NnRmsNormOpConfig{invRmsBufferIndex, 1});
        if (yBufferIndex != yqBufferIndex) {
            end.addOp(OP_CAST, "final_cast_y", 0,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex), pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                size0(), NnCastOpCodeConfig{});
        }
        end.addOp(OP_MATMUL, "final_matmul_logits", 0,
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex), pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            size2D(h->weightType, n.wclsSlice.n, n.wclsSlice.d0),
            NnMatmulOpConfig{0, 0, unusedExpertIndexesBufferIndex});
        end.addOp(OP_CAST, "final_cast_logits", 0,
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, n.outputPipeIndex), size0(), NnCastOpCodeConfig{});
        end.addSync(n.outputPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);
        nodeBuilder.addSegment(end.build());
        n.nodeConfigs[nodeIndex] = nodeBuilder.build();
    }
    return std::unique_ptr<LlmGraph>(graph.release());
}

void Qwen35LlmModel::loadGraph(
    const char *path, LlmGraph *baseGraph, NnRootWeightLoader *loader) const {
    Qwen35Graph *graph = static_cast<Qwen35Graph *>(baseGraph);
    const Qwen35LlmHeader *h = static_cast<const Qwen35LlmHeader *>(graph->header);
    const NnUint nNodes = graph->netConfig.nNodes;
    MmapFile file;
    openMmapFile(&file, path, h->fileSize);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    // CPU ops retain pointers into the mapping in this debug mode.
    assert(nNodes == 1u);
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
#endif
    printf("💿 Loading weights...\n");
    NnSize offset = h->headerSize;
    const auto take = [&](NnSize nBytes) -> NnByte * {
        if (offset > h->fileSize || nBytes > h->fileSize - offset)
            throw std::runtime_error("Truncated weight file at byte " + std::to_string(offset));
        NnByte *weight = static_cast<NnByte *>(file.data) + offset;
        offset += nBytes;
        return weight;
    };
    const auto row = [&](const char *name, NnUint layer, NnRowMatmulSlice &slice) {
        loader->loadRowMatmulSlices(name, layer, 0u, &slice, take(slice.size.nBytes));
    };
    const auto col = [&](const char *name, NnUint layer, NnColMatmulSlice &slice) {
        loader->loadColMatmulSlices(name, layer, 0u, &slice, take(slice.size.nBytes));
    };
    const auto all = [&](const char *name, NnUint layer, NnSize nBytes) {
        loader->loadAll(name, layer, nBytes, take(nBytes));
    };

    // Convolutions store one contiguous kernel per channel, so row slicing
    // selects exactly the channels owned by each node's Q/K/V projections.
    NnRowMatmulSlice convKey = sliceRowMatmul(F_32, nNodes, h->linearConvKernelDim,
        h->linearNKeyHeads * h->linearKeyHeadDim);
    NnRowMatmulSlice convValue = sliceRowMatmul(F_32, nNodes, h->linearConvKernelDim,
        h->linearNValueHeads * h->linearValueHeadDim);
    // The converter writes consecutive [A_log, dt_bias] vectors. Treat them
    // as two rows and slice columns to pack [local A_log, local dt_bias].
    NnColMatmulSlice delta = sliceColMatmul(F_32, nNodes, h->linearNValueHeads, 2u);

    loader->loadRoot("embedding", 0u, graph->tokenEmbeddingSize.nBytes,
        take(graph->tokenEmbeddingSize.nBytes));
    for (NnUint layer = 0u; layer < h->nLayers; layer++) {
        if (isFullAttentionLayer(h, layer)) {
            row("block_matmul_q", layer, graph->qSlice);
            row("block_matmul_gate", layer, graph->gateSlice);
            row("block_matmul_k", layer, graph->kSlice);
            row("block_matmul_v", layer, graph->vSlice);
            col("block_matmul_wo", layer, graph->woSlice);
            all("block_norm_q", layer, graph->qkRmsNormSize.nBytes);
            all("block_norm_k", layer, graph->qkRmsNormSize.nBytes);
        } else {
            row("block_linear_matmul_q", layer, graph->linearQSlice);
            row("block_linear_matmul_k", layer, graph->linearKSlice);
            row("block_linear_matmul_v", layer, graph->linearVSlice);
            row("block_linear_matmul_z", layer, graph->linearZSlice);
            row("block_linear_matmul_a", layer, graph->linearASlice);
            row("block_linear_matmul_b", layer, graph->linearBSlice);
            row("block_linear_conv_q", layer, convKey);
            row("block_linear_conv_k", layer, convKey);
            row("block_linear_conv_v", layer, convValue);
            col("block_linear_delta", layer, delta);
            all("block_linear_norm", layer, size1D(F_32, h->linearValueHeadDim).nBytes);
            col("block_matmul_wo", layer, graph->linearOutSlice);
        }
        row("block_matmul_w1", layer, graph->w1Slice);
        col("block_matmul_w2", layer, graph->w2Slice);
        row("block_matmul_w3", layer, graph->w3Slice);
        all("block_norm_0", layer, graph->rmsNormSize.nBytes);
        all("block_norm_1", layer, graph->rmsNormSize.nBytes);
    }
    all("final_norm", 0u, graph->rmsNormSize.nBytes);
    row("final_matmul_logits", 0u, graph->wclsSlice);
    if (offset != h->fileSize)
        throw std::runtime_error("Unexpected trailing bytes in Qwen3.5 weight file: " +
            std::to_string(h->fileSize - offset));
    loader->finish();
    printf("💿 Qwen3.5 weights loaded\n");
}

const LlmModel &getQwen35LlmModel() {
    static Qwen35LlmModel model;
    return model;
}
