#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <cerrno>
#include <stdexcept>

static const char *hiddenActToString(LlmHiddenAct act) {
    if (act == HIDDEN_ACT_GELU) return "Gelu";
    if (act == HIDDEN_ACT_SILU) return "Silu";
    throw std::runtime_error("Unsupported hidden act");
}

static const char *ropeTypeToString(NnRopeType type) {
    if (type == ROPE_LLAMA) return "Llama";
    if (type == ROPE_LLAMA3_1) return "Llama3.1";
    if (type == ROPE_FALCON) return "Falcon";
    throw std::runtime_error("Unsupported rope type");
}

static const char *archTypeToString(LlmArchType type) {
    if (type == LLAMA) return "Llama";
    if (type == QWEN3) return "Qwen3";
    if (type == QWEN3_MOE) return "Qwen3 MoE";
    throw std::runtime_error("Unsupported architecture");
}

static float convertNormEpsilon(int value) {
    if (value == 5) return 1e-05f;
    if (value == 6) return 1e-06f;
    throw std::runtime_error("Unsupported norm epsilon");
}

LlmHeader loadLlmHeader(const char *path, const NnUint maxSeqLen, NnFloatType syncType) {
    LlmHeader header;
    std::memset(&header, 0, sizeof(LlmHeader));
    header.weightType = F_UNK;
    header.hiddenAct = HIDDEN_ACT_SILU;
    header.ropeType = ROPE_LLAMA;
    header.ropeTheta = 10000.0f;
    header.ropeScalingFactor = 1.0f;
    header.normEpsilon = 1e-5f;
    header.moeHiddenDim = 0u;

    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *fd = fdPtr.get();
    if (fd == NULL)
        throw std::runtime_error(std::string("Cannot open model file (") + path + std::string("): ") + std::strerror(errno));

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read magic value");

    if (magic == 0xABCD00 || magic == 0xABCD01)
        throw std::runtime_error("Old model format is not supported");
    if (magic != 0xA00ABCD)
        throw std::runtime_error("Unsupported magic number");

    if (fread(&header.headerSize, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read header size");

    std::vector<int> bufferPtr(header.headerSize);
    int *buffer = &bufferPtr[0];
    if (fread(buffer, header.headerSize, 1, fd) != 1)
        throw std::runtime_error("Cannot read header values");

    int nKv = (header.headerSize - 2 * sizeof(int)) / sizeof(int);

    for (int i = 0; i < nKv; i += 2) {
        int key = buffer[i];
        int value = buffer[i + 1];
        if (key == VERSION) header.version = value;
        else if (key == ARCH_TYPE) header.archType = (LlmArchType)value;
        else if (key == DIM) header.dim = value;
        else if (key == HIDDEN_DIM) header.hiddenDim = value;
        else if (key == N_LAYERS) header.nLayers = value;
        else if (key == N_HEADS) header.nHeads = value;
        else if (key == N_KV_HEADS) header.nKvHeads = value;
        else if (key == N_EXPERTS) header.nExperts = value;
        else if (key == N_ACTIVE_EXPERTS) header.nActiveExperts = value;
        else if (key == VOCAB_SIZE) header.vocabSize = value;
        else if (key == SEQ_LEN) header.seqLen = value;
        else if (key == HIDDEN_ACT) header.hiddenAct = (LlmHiddenAct)value;
        else if (key == ROPE_THETA) header.ropeTheta = (float)value;
        else if (key == WEIGHT_FLOAT_TYPE) header.weightType = (NnFloatType)value;
        else if (key == ROPE_SCALING_FACTOR) header.ropeScalingFactor = (float)value;
        else if (key == ROPE_SCALING_LOW_FREQ_FACTOR) header.ropeScalingLowFreqFactor = (float)value;
        else if (key == ROPE_SCALING_HIGH_FREQ_FACTORY) header.ropeScalingHighFreqFactory = (float)value;
        else if (key == ROPE_SCALING_ORIG_MAX_SEQ_LEN) header.ropeScalingOrigMaxSeqLen = value;
        else if (key == ROPE_TYPE) header.ropeType = (NnRopeType)value;
        else if (key == HEAD_DIM) header.headDim = value;
        else if (key == NORM_EPSILON) header.normEpsilon = convertNormEpsilon(value);
        else if (key == MOE_HIDDEN_DIM) header.moeHiddenDim = value;
        else throw std::runtime_error("Unsupported header key");
    }

    if (header.weightType == F_UNK)
        throw std::runtime_error("Model does not specify weight type");

    header.origSeqLen = header.seqLen;
    if (maxSeqLen > 0 && header.seqLen > maxSeqLen)
        header.seqLen = maxSeqLen;

    if (header.headDim == 0)
        header.headDim = header.dim / header.nHeads;
    header.qDim = header.headDim * header.nHeads;
    header.kvDim = header.headDim * header.nKvHeads;
    header.syncType = syncType;
    header.fileSize = (NnSize)seekToEnd(fd);

    if (header.archType == QWEN3 || header.archType == QWEN3_MOE)
        header.ropeType = ROPE_FALCON;
    return header;
}

void printLlmHeader(LlmHeader *header) {
    printf("💡 Arch: %s\n", archTypeToString(header->archType));
    printf("💡 HiddenAct: %s\n", hiddenActToString(header->hiddenAct));
    printf("💡 Dim: %u\n", header->dim);
    printf("💡 HeadDim: %u\n", header->headDim);
    printf("💡 QDim: %u\n", header->qDim);
    printf("💡 KvDim: %u\n", header->kvDim);
    printf("💡 HiddenDim: %u\n", header->hiddenDim);
    printf("💡 VocabSize: %u\n", header->vocabSize);
    printf("💡 nLayers: %u\n", header->nLayers);
    printf("💡 nHeads: %u\n", header->nHeads);
    printf("💡 nKvHeads: %u\n", header->nKvHeads);
    if (header->seqLen != header->origSeqLen) {
        printf("💡 OrigSeqLen: %u\n", header->origSeqLen);
    }
    if (header->nExperts > 0) {
        printf("💡 nExperts: %u\n", header->nExperts);
        printf("💡 nActiveExperts: %u\n", header->nActiveExperts);
        printf("💡 MoeHiddenDim: %u\n", header->moeHiddenDim);
    }
    printf("💡 SeqLen: %u\n", header->seqLen);
    printf("💡 NormEpsilon: %f\n", header->normEpsilon);
    printf("💡 RopeType: %s\n", ropeTypeToString(header->ropeType));
    printf("💡 RopeTheta: %.0f\n", header->ropeTheta);
    if (header->ropeType == ROPE_LLAMA3_1) {
        printf("💡 RopeScaling: f=%.1f, l=%.1f, h=%.1f, o=%d\n",
            header->ropeScalingFactor,
            header->ropeScalingLowFreqFactor,
            header->ropeScalingHighFreqFactory,
            header->ropeScalingOrigMaxSeqLen);
    }
}

LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches) {
    NnUint nExpertsOr1 = std::max(h->nExperts, 1u);
    NnUint nActiveExpertsOr1 = std::max(h->nActiveExperts, 1u);
    NnUint ffDim = h->hiddenDim;

    if (h->archType == QWEN3_MOE)
        ffDim = h->moeHiddenDim;

    LlmNet n;
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);
    n.qkRmsNormSize = size1D(F_32, h->headDim);
    n.moeGateSize = size2D(F_32, h->dim, h->nExperts);

    NnKvCacheSlice kvCacheSlice = sliceKvCache(h->kvDim, h->seqLen, nNodes);
    NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(h->nHeads, h->seqLen, nNodes, nBatches);

    n.qSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->qDim);
    n.kSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.vSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.woSlice = sliceColMatmul(h->weightType, nNodes, h->qDim, h->dim);

    n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
    n.w2Slice = sliceColMatmul(h->weightType, nNodes, ffDim, h->dim);
    n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, ffDim);
    n.wclsSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->vocabSize);
 
    NnUint nQNormColumns = 1;
    NnUint nKNormColumns = 1;
    NnUint nInvBufferColumns = 1;
    if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
        ASSERT_EQ(n.qSlice.d0 % h->headDim, 0);
        ASSERT_EQ(n.kSlice.d0 % h->headDim, 0);
        nQNormColumns = n.qSlice.d0 / h->headDim;
        nKNormColumns = n.kSlice.d0 / h->headDim;
        nInvBufferColumns = std::max(nQNormColumns, nKNormColumns);
    }

    NnNetConfigBuilder netBuilder(nNodes, nBatches);

    n.positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n.tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n.xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n.logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    const NnUint zqPipeIndex = netBuilder.addPipe("ZQ", size2D(h->syncType, nBatches, h->dim * nNodes));

    netBuilder.addPreSync(n.positionPipeIndex);

    n.header = h;
    n.netConfig = netBuilder.build();
    n.nodeConfigs = new NnNodeConfig[nNodes];

    for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        NnRopeSlice ropeSlice = sliceRope(h->ropeType, h->qDim, h->kvDim, h->nKvHeads, nNodes, h->seqLen, h->headDim, h->ropeTheta, nodeIndex);
        NnNodeConfigBuilder nodeBuilder(nodeIndex);

        const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
        const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
        const NnUint yqBufferIndex = h->syncType == F_32
            ? yBufferIndex
            : nodeBuilder.addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));

        const NnUint zBufferIndex = nodeBuilder.addBuffer("z", size2D(F_32, nBatches, h->qDim));
        const NnUint zqSliceBufferIndex = nodeBuilder.addBuffer("q_z_slice", size2D(h->syncType, nBatches, h->qDim / nNodes));

        const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, n.qSlice.d0));
        const NnUint kTempBufferIndex = nodeBuilder.addBuffer("k_temp", size2D(F_32, nBatches, n.kSlice.d0));
        const NnUint vTempBufferIndex = nodeBuilder.addBuffer("v_temp", size2D(F_32, nBatches, n.vSlice.d0));

        const NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, nBatches, nInvBufferColumns));

        const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", ropeSlice.cacheSize);
        const NnUint attBufferIndex = nodeBuilder.addBuffer("att", multiHeadAttSlice.attSize);
        const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer("lg", size2D(F_32, nBatches, h->vocabSize / nNodes));

        // not moe
        const NnUint dBufferIndex = nodeBuilder.addBuffer("d", size2D(F_32, nBatches, n.w1Slice.d0));
        const NnUint dqBufferIndex = h->syncType == F_32
            ? dBufferIndex
            : nodeBuilder.addBuffer("q_d", size2D(h->syncType, nBatches, n.w1Slice.d0));
        const NnUint lBufferIndex = nodeBuilder.addBuffer("l", size2D(F_32, nBatches, n.w3Slice.d0));

        // moe
        const NnUint moeGtBufferIndex = nodeBuilder.addBuffer("gt", size2D(F_32, nBatches, nExpertsOr1));
        const NnUint moeExpertIndexesBufferIndex = nodeBuilder.addBuffer("act_exp_ix", size2D(F_32, nBatches, nActiveExpertsOr1));
        const NnUint moeYBufferIndex = nodeBuilder.addBuffer("moe_y", size3D(F_32, nActiveExpertsOr1, nBatches, h->dim));
        const NnUint moeYqBufferIndex = h->syncType == F_32
            ? moeYBufferIndex
            : nodeBuilder.addBuffer("q_moe_y", size3D(h->syncType, nActiveExpertsOr1, nBatches, h->dim));
        const NnUint moeDBufferIndex = nodeBuilder.addBuffer("moe_d", size3D(F_32, nActiveExpertsOr1, nBatches, n.w1Slice.d0));
        const NnUint moeDQBufferIndex = h->syncType == F_32
            ? moeDBufferIndex
            : nodeBuilder.addBuffer("q_moe_d", size3D(h->syncType, nActiveExpertsOr1, nBatches, n.w1Slice.d0));
        const NnUint moeLBufferIndex = nodeBuilder.addBuffer("moe_l", size3D(F_32, nActiveExpertsOr1, nBatches, n.w3Slice.d0));
        const NnUint moeSBufferIndex = nodeBuilder.addBuffer("moe_s", size3D(F_32, nActiveExpertsOr1, nBatches, 1));

        NnSegmentConfigBuilder start;
        if (nodeIndex == 0) {
            start.addOp(
                OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, n.tokenPipeIndex),
                pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                n.tokenEmbeddingSize,
                NnEmbeddingOpConfig{});
        }
        start.addSync(n.xPipeIndex, SYNC_WITH_ROOT);
        nodeBuilder.addSegment(start.build());

        for (NnUint layerIndex = 0; layerIndex < h->nLayers; layerIndex++) {
            const NnUint kBufferIndex = nodeBuilder.addBuffer("k", kvCacheSlice.keySize);
            const NnUint vBufferIndex = nodeBuilder.addBuffer("v", kvCacheSlice.valueSize);

            NnSegmentConfigBuilder att;
            NnSegmentConfigBuilder ff;

            // att
            if (layerIndex == 0) {
                att.addOp(
                    OP_CAST, "block_cast_x", layerIndex,
                    pointerBatchConfig(SRC_PIPE, n.xPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            } else {
                att.addOp(
                    OP_MERGE_ADD, "block_merge_add", layerIndex,
                    pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                    pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                    size0(),
                    NnMergeAddOpCodeConfig{});
            }

            att.addOp(
                OP_INV_RMS, "block_norm_pre_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            att.addOp(
                OP_RMS_NORM, "block_norm_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
            if (yBufferIndex != yqBufferIndex) {
                att.addOp(
                    OP_CAST, "block_cast_y", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    size0(),
                    NnCastOpCodeConfig{});
            }
            att.addOp(
                OP_MATMUL, "block_matmul_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size2D(h->weightType, n.qSlice.n, n.qSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            att.addOp(
                OP_MATMUL, "block_matmul_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size2D(h->weightType, n.kSlice.n, n.kSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            att.addOp(
                OP_MATMUL, "block_matmul_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                size2D(h->weightType, n.vSlice.n, n.vSlice.d0),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});

            if (h->archType == QWEN3 || h->archType == QWEN3_MOE) {
                att.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nQNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_q", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nQNormColumns});

                att.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                    size0(),
                    NnInvRmsOpConfig{h->normEpsilon, nKNormColumns});
                att.addOp(
                    OP_RMS_NORM, "block_norm_k", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                    size2D(F_32, 1, n.header->headDim),
                    NnRmsNormOpConfig{invRmsBufferIndex, nKNormColumns});
            }

            att.addOp(
                OP_ROPE, "block_rope_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 1, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_ROPE, "block_rope_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size0(),
                NnRopeOpConfig{n.header->ropeType, 0, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_SHIFT, "block_shift_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, kBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex});
            att.addOp(
                OP_SHIFT, "block_shift_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                pointerRawConfig(SRC_BUFFER, vBufferIndex),
                size0(),
                NnShiftOpCodeConfig{n.positionPipeIndex});
            att.addOp(
                OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                size0(),
                NnMultiHeadAttOpConfig{
                    multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
                    h->nKvHeads, h->headDim, h->seqLen, n.qSlice.d0, kvCacheSlice.kvDim0,
                    n.positionPipeIndex, qBufferIndex, kBufferIndex, vBufferIndex, attBufferIndex});
            att.addOp(
                OP_CAST, "block_cast_y2", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex),
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_wo", layerIndex,
                pointerBatchConfig(SRC_BUFFER, zqSliceBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                size2D(h->weightType, n.woSlice.n0, n.woSlice.d),
                NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            att.addOp(
                OP_CAST, "block_cast_d", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addSync(zqPipeIndex, SYNC_NODE_SLICES);

            // ff
            ff.addOp(
                OP_MERGE_ADD, "block_merge_add2", layerIndex,
                pointerBatchConfig(SRC_PIPE, zqPipeIndex),
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
            ff.addOp(
                OP_INV_RMS, "block_norm_pre_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon, 1});
            ff.addOp(
                OP_RMS_NORM, "block_norm_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex, 1});

            if (h->archType == QWEN3_MOE) {
                ff.addOp(
                    OP_REPEAT_Z, "block_moe_y_repeat", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    size0(),
                    NnRepeatZOpCodeConfig{});
                ff.addOp(
                    OP_MATMUL, "block_moe_gate", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    n.moeGateSize,
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SOFTMAX, "block_moe_softmax", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    size0(),
                    NnSoftmaxOpCodeConfig{});
                ff.addOp(
                    OP_MOE_GATE, "block_moe_gate2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeGtBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeSBufferIndex),
                    size0(),
                    NnMoeGateOpCodeConfig{h->nActiveExperts, 1u, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w1", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w1Slice.n, n.w1Slice.d0),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeLBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w3Slice.n, n.w3Slice.d0),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SILU, "block_act", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size0(),
                    NnSiluOpCodeConfig{});
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{moeLBufferIndex});
                if (moeDBufferIndex != moeDQBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_d2", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, moeDBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeDQBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    size3D(h->weightType, h->nExperts, n.w2Slice.n0, n.w2Slice.d),
                    NnMatmulOpConfig{h->nExperts, h->nActiveExperts, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SCALE, "block_moe_scale", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    size0(),
                    NnScaleOpCodeConfig{moeSBufferIndex});
                ff.addOp(
                    OP_MERGE_SUM, "block_moe_merge_sum", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, moeYBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    size0(),
                    NnMergeSumOpCodeConfig{});
            } else {
                if (yBufferIndex != yqBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_y3", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w1", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size2D(h->weightType, n.w1Slice.n, n.w1Slice.d0),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_MATMUL, "block_matmul_w3", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, lBufferIndex),
                    size2D(h->weightType, n.w3Slice.n, n.w3Slice.d0),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
                ff.addOp(
                    OP_SILU, "block_act", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size0(),
                    NnSiluOpCodeConfig{});
                ff.addOp(
                    OP_MUL, "block_mul", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                    size0(),
                    NnMulOpCodeConfig{lBufferIndex});
                if (dBufferIndex != dqBufferIndex) {
                    ff.addOp(
                        OP_CAST, "block_cast_d2", layerIndex,
                        pointerBatchConfig(SRC_BUFFER, dBufferIndex),
                        pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                        size0(),
                        NnCastOpCodeConfig{});
                }
                ff.addOp(
                    OP_MATMUL, "block_matmul_w2", layerIndex,
                    pointerBatchConfig(SRC_BUFFER, dqBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                    size2D(h->weightType, n.w2Slice.n0, n.w2Slice.d),
                    NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
            }
            ff.addOp(
                OP_CAST, "block_cast_d3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
            ff.addSync(zqPipeIndex, SYNC_NODE_SLICES);

            nodeBuilder.addSegment(att.build());
            nodeBuilder.addSegment(ff.build());
        }

        NnSegmentConfigBuilder end;
        end.addOp(
            OP_MERGE_ADD, "final_merge_add", 0,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
        end.addOp(
            OP_INV_RMS, "final_norm_pre", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
            size0(),
            NnInvRmsOpConfig{h->normEpsilon, 1});
        end.addOp(
            OP_RMS_NORM, "final_norm", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, yBufferIndex),
            n.rmsNormSize,
            NnRmsNormOpConfig{invRmsBufferIndex, 1});
        if (yBufferIndex != yqBufferIndex) {
            end.addOp(
                OP_CAST, "final_cast_y", 0,
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                size0(),
                NnCastOpCodeConfig{});
        }
        end.addOp(
            OP_MATMUL, "final_matmul_logits", 0,
            pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            size2D(h->weightType, n.wclsSlice.n, n.wclsSlice.d0),
            NnMatmulOpConfig{});
        end.addOp(
            OP_CAST, "final_cast_logits", 0,
            pointerBatchConfig(SRC_BUFFER, logitsSliceBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, n.logitsPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
        end.addSync(n.logitsPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);

        nodeBuilder.addSegment(end.build());
        n.nodeConfigs[nodeIndex] = nodeBuilder.build();
    }
    return n;
}

void releaseLlmNet(LlmNet *net) {
    for (NnUint nodeIndex = 0u; nodeIndex < net->netConfig.nNodes; nodeIndex++)
        releaseNodeConfig(&net->nodeConfigs[nodeIndex]);
    releaseNetConfig(&net->netConfig);
    delete[] net->nodeConfigs;
}

void loadLlmNetWeight(const char *path, LlmNet *net, NnRootWeightLoader *loader) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(net->netConfig.nNodes == 1u);
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...\n");
#endif

    Timer timer;
    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);

    for (NnUint layerIndex = 0u; layerIndex < net->header->nLayers; layerIndex++) {
        b += loader->loadRowMatmulSlices("block_matmul_q", layerIndex, 0u, &net->qSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_k", layerIndex, 0u, &net->kSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_v", layerIndex, 0u, &net->vSlice, b);
        b += loader->loadColMatmulSlices("block_matmul_wo", layerIndex, 0u, &net->woSlice, b);

        if (net->header->nExperts > 0u) {
            b += loader->loadAll("block_moe_gate", layerIndex, net->moeGateSize.nBytes, b);
            for (NnUint expertIndex = 0u; expertIndex < net->header->nExperts; expertIndex++) {
                b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, expertIndex, &net->w1Slice, b);
                b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, expertIndex, &net->w2Slice, b);
                b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, expertIndex, &net->w3Slice, b);
            }
        } else {
            b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, 0u, &net->w1Slice, b);
            b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, 0u, &net->w2Slice, b);
            b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, 0u, &net->w3Slice, b);
        }

        if (net->header->archType == QWEN3 || net->header->archType == QWEN3_MOE) {
            b += loader->loadAll("block_norm_q", layerIndex, net->qkRmsNormSize.nBytes, b);
            b += loader->loadAll("block_norm_k", layerIndex, net->qkRmsNormSize.nBytes, b);
        }

        b += loader->loadAll("block_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
        b += loader->loadAll("block_norm_1", layerIndex, net->rmsNormSize.nBytes, b);

        if (timer.elapsedMiliseconds() > 10000)
            printf("💿 Loaded %u/%u\n", layerIndex + 1, net->header->nLayers);
    }

    b += loader->loadAll("final_norm", 0u, net->rmsNormSize.nBytes, b);
    b += loader->loadRowMatmulSlices("final_matmul_logits", 0u, 0u, &net->wclsSlice, b);

    long long missingBytes = (long long)(b - data) - net->header->fileSize;
    if (missingBytes != 0u)
        throw std::runtime_error("Missing bytes in weight file: " + std::to_string(missingBytes));
    printf("💿 Weights loaded\n");

    loader->finish();
}
