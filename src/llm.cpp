#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "mmap.hpp"
#include "llm.hpp"
#include <stdexcept>

static const char *hiddenActToString(LlmHiddenAct act) {
    if (act == HIDDEN_ACT_GELU) return "Gelu";
    if (act == HIDDEN_ACT_SILU) return "Silu";
    throw std::runtime_error("Unsupported hidden act");
}

static const char *ropeTypeToString(NnRopeType type) {
    if (type == ROPE_LLAMA) return "Llama";
    if (type == ROPE_LLAMA3_1) return "Llama3.1";
    throw std::runtime_error("Unsupported rope type");
}

static const char *archTypeToString(LlmArchType type) {
    if (type == LLAMA) return "Llama";
    throw std::runtime_error("Unsupported architecture");
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

    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *fd = fdPtr.get();
    if (fd == NULL)
        throw std::runtime_error("Cannot open model file");

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

    int nKv = (header.headerSize - 2  *sizeof(int)) / sizeof(int);

    NnFloatType modelWeightType = F_UNK;
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
        else throw std::runtime_error("Unsupported header key");
    }

    if (header.weightType == F_UNK)
        throw std::runtime_error("Model does not specify weight type");

    header.origSeqLen = header.seqLen;
    if (maxSeqLen > 0 && header.seqLen > maxSeqLen)
        header.seqLen = maxSeqLen;

    header.headSize = header.dim / header.nHeads;
    header.kvDim = (header.dim  *header.nKvHeads) / header.nHeads;
    header.syncType = syncType;
    header.fileSize = (NnSize)seekToEnd(fd);
    return header;
}

void printLlmHeader(LlmHeader *header) {
    printf("💡 Arch: %s\n", archTypeToString(header->archType));
    printf("💡 HiddenAct: %s\n", hiddenActToString(header->hiddenAct));
    printf("💡 Dim: %u\n", header->dim);
    printf("💡 KvDim: %u\n", header->kvDim);
    printf("💡 HiddenDim: %u\n", header->hiddenDim);
    printf("💡 VocabSize: %u\n", header->vocabSize);
    printf("💡 nLayers: %u\n", header->nLayers);
    printf("💡 nHeads: %u\n", header->nHeads);
    printf("💡 nKvHeads: %u\n", header->nKvHeads);
    if (header->seqLen != header->origSeqLen) {
        printf("💡 OrigSeqLen: %u\n", header->origSeqLen);
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
    LlmNet n;
    n.tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n.rmsNormSize = size1D(F_32, h->dim);

    NnKvCacheSlice kvCacheSlice = sliceKvCache(h->kvDim, h->seqLen, nNodes);
    NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(h->nHeads, h->seqLen, nNodes, nBatches);

    n.qSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->dim);
    n.kSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.vSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->kvDim);
    n.woSlice = sliceColMatmul(h->weightType, nNodes, h->dim, h->dim);

    n.w1Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->hiddenDim);
    n.w2Slice = sliceColMatmul(h->weightType, nNodes, h->hiddenDim, h->dim);
    n.w3Slice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->hiddenDim);
    n.wclsSlice = sliceRowMatmul(h->weightType, nNodes, h->dim, h->vocabSize);

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
        NnRopeSlice ropeSlice = sliceRope(h->dim, h->kvDim, h->nKvHeads, nNodes, h->seqLen, h->headSize, h->ropeTheta, nodeIndex);
        NnNodeConfigBuilder nodeBuilder(nodeIndex);

        const NnUint xBufferIndex = nodeBuilder.addBuffer("x", size2D(F_32, nBatches, h->dim));
        
        const NnUint yBufferIndex = nodeBuilder.addBuffer("y", size2D(F_32, nBatches, h->dim));
        const NnUint yqBufferIndex = h->syncType == F_32
            ? yBufferIndex
            : nodeBuilder.addBuffer("yq", size2D(h->syncType, nBatches, h->dim));
        const NnUint yqSliceIndex = nodeBuilder.addBuffer("yq_slice", size2D(h->syncType, nBatches, h->dim / nNodes));

        const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, nBatches, n.qSlice.d0));
        const NnUint kTempBufferIndex = nodeBuilder.addBuffer("k_temp", size2D(F_32, nBatches, n.kSlice.d0));
        const NnUint vTempBufferIndex = nodeBuilder.addBuffer("v_temp", size2D(F_32, nBatches, n.vSlice.d0));

        const NnUint dBufferIndex = nodeBuilder.addBuffer("d", size2D(F_32, nBatches, n.w1Slice.d0));
        const NnUint dqBufferIndex = h->syncType == F_32
            ? dBufferIndex
            : nodeBuilder.addBuffer("d", size2D(h->syncType, nBatches, n.w1Slice.d0));
        const NnUint lBufferIndex = nodeBuilder.addBuffer("l", size2D(F_32, nBatches, n.w3Slice.d0));
        const NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, nBatches, 1));
        const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", ropeSlice.cacheSize);
        const NnUint attBufferIndex = nodeBuilder.addBuffer("att", multiHeadAttSlice.attSize);
        const NnUint logitsSliceBufferIndex = nodeBuilder.addBuffer("lg", size2D(F_32, nBatches, h->vocabSize / nNodes));

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
                OP_INV_RMS, "block_inv_rms_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon});
            att.addOp(
                OP_RMS_NORM, "block_rms_norm_0", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex});
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
                NnMatmulOpConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size2D(h->weightType, n.kSlice.n, n.kSlice.d0),
                NnMatmulOpConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_v", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, vTempBufferIndex),
                size2D(h->weightType, n.vSlice.n, n.vSlice.d0),
                NnMatmulOpConfig{});

            att.addOp(
                OP_ROPE_LLAMA, "block_rope_q", layerIndex,
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                pointerBatchConfig(SRC_BUFFER, qBufferIndex),
                size0(),
                NnRopeLlamaOpConfig{true, n.positionPipeIndex, ropeCacheBufferIndex, 
                    h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
                    ropeSlice});
            att.addOp(
                OP_ROPE_LLAMA, "block_rope_k", layerIndex,
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                pointerBatchConfig(SRC_BUFFER, kTempBufferIndex),
                size0(),
                NnRopeLlamaOpConfig{false, n.positionPipeIndex, ropeCacheBufferIndex, 
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
                pointerBatchedSliceConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchedSliceConfig(SRC_BUFFER, yBufferIndex),
                size0(),
                NnMultiHeadAttOpConfig{
                    multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
                    h->nKvHeads, h->headSize, h->seqLen, n.qSlice.d0, kvCacheSlice.kvDim0,
                    n.positionPipeIndex, qBufferIndex, kBufferIndex, vBufferIndex, attBufferIndex});
            att.addOp(
                OP_CAST, "block_cast_y2", layerIndex,
                pointerBatchedSliceConfig(SRC_BUFFER, yBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yqSliceIndex),
                size0(),
                NnCastOpCodeConfig{});
            att.addOp(
                OP_MATMUL, "block_matmul_wo", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqSliceIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                size2D(h->weightType, n.woSlice.n0, n.woSlice.d),
                NnMatmulOpConfig{});
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
                OP_INV_RMS, "block_inv_rms_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{h->normEpsilon});
            ff.addOp(
                OP_RMS_NORM, "block_rms_norm_1", layerIndex,
                pointerBatchConfig(SRC_BUFFER, xBufferIndex),
                pointerBatchConfig(SRC_BUFFER, yBufferIndex),
                n.rmsNormSize,
                NnRmsNormOpConfig{invRmsBufferIndex});
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
                NnMatmulOpConfig{});
            ff.addOp(
                OP_MATMUL, "block_matmul_w3", layerIndex,
                pointerBatchConfig(SRC_BUFFER, yqBufferIndex),
                pointerBatchConfig(SRC_BUFFER, lBufferIndex),
                size2D(h->weightType, n.w3Slice.n, n.w3Slice.d0),
                NnMatmulOpConfig{});
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
                NnMatmulOpConfig{});
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
            OP_INV_RMS, "final_inv_rms", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
            size0(),
            NnInvRmsOpConfig{h->normEpsilon});
        end.addOp(
            OP_RMS_NORM, "final_rms_norm", 0,
            pointerBatchConfig(SRC_BUFFER, xBufferIndex),
            pointerBatchConfig(SRC_BUFFER, yBufferIndex),
            n.rmsNormSize,
            NnRmsNormOpConfig{invRmsBufferIndex});
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
    for (NnUint nodeIndex = 0; nodeIndex < net->netConfig.nNodes; nodeIndex++)
        releaseNodeConfig(&net->nodeConfigs[nodeIndex]);
    releaseNetConfig(&net->netConfig);
    delete[] net->nodeConfigs;
}

void loadLlmNetWeight(const char *path, LlmNet *net, NnRootWeightLoader *loader) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
#if DEBUG_USE_MMAP_FOR_WEIGHTS
    assert(net->netConfig.nNodes == 1);
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...\n");
#endif

    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    NnUint nodeIndex = 0;
    b += loader->loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);

    for (NnUint layerIndex = 0; layerIndex < net->header->nLayers; layerIndex++) {
        b += loader->loadRowMatmulSlices("block_matmul_q", layerIndex, &net->qSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_k", layerIndex, &net->kSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_v", layerIndex, &net->vSlice, b);
        b += loader->loadColMatmulSlices("block_matmul_wo", layerIndex, &net->woSlice, b);
        b += loader->loadRowMatmulSlices("block_matmul_w1", layerIndex, &net->w1Slice, b);
        b += loader->loadColMatmulSlices("block_matmul_w2", layerIndex, &net->w2Slice, b);
        b += loader->loadRowMatmulSlices("block_matmul_w3", layerIndex, &net->w3Slice, b);
        b += loader->loadAll("block_rms_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
        b += loader->loadAll("block_rms_norm_1", layerIndex, net->rmsNormSize.nBytes, b);
    }

    b += loader->loadAll("final_rms_norm", 0, net->rmsNormSize.nBytes, b);
    b += loader->loadRowMatmulSlices("final_matmul_logits", 0, &net->wclsSlice, b);

    long long missingBytes = (long long)(b - data) - net->header->fileSize;
    if (missingBytes != 0)
        throw std::runtime_error("Missing bytes in weight file: " + std::to_string(missingBytes));
    printf("💿 Weights loaded\n");

    loader->finish();
}
