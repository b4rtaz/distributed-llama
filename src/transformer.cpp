#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <string.h>
#include "utils.hpp"
#include "socket.hpp"
#include "commands.hpp"
#include "transformer.hpp"

#define IS_ROOT_SLICE(sliceIndex) (sliceIndex == 0)

TransformerSpec Transformer::loadSpecFromFile(const char* path, const unsigned int nSlices, const unsigned int maxSeqLen, FloatType weightsFloatType, FloatType bufferFloatType) {
    TransformerSpec spec;
    memset(&spec, 0, sizeof(TransformerSpec));
    spec.hiddenAct = SILU;
    spec.ropeType = ROPE_UNKNOWN;
    spec.ropeTheta = 10000.0f;

    FILE* fd = fopen(path, "rb");
    if (fd == NULL) {
        throw std::runtime_error("Cannot open model file");
    }

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1) {
        throw std::runtime_error("Cannot read magic value");
    }
    if (magic == 0xABCD00 || magic == 0xABCD01) {
        TransformerFileOldHeader header;
        if (fread(&header, sizeof(header), 1, fd) != 1) {
            throw std::runtime_error("Cannot read header");
        }
        spec.headerSize = sizeof(int) + sizeof(TransformerFileOldHeader);
        spec.archType = (TransformerArchType)magic;
        spec.dim = header.dim;
        spec.hiddenDim = header.hiddenDim;
        spec.nLayers = header.nLayers;
        spec.nHeads = header.nHeads;
        spec.nKvHeads = header.nKvHeads;
        spec.nExperts = header.nExperts;
        spec.nActiveExperts = header.nActiveExperts;
        spec.vocabSize = header.vocabSize;
        spec.seqLen = header.seqLen;
    } else if (magic == 0xA00ABCD) {
        if (fread(&spec.headerSize, sizeof(int), 1, fd) != 1) {
            throw std::runtime_error("Cannot read header size");
        }
        int buffer[spec.headerSize];
        if (fread(&buffer, spec.headerSize, 1, fd) != 1) {
            throw std::runtime_error("Cannot read header values");
        }
        int nKv = (spec.headerSize - 2 * sizeof(int)) / sizeof(int);

        FloatType modelWeightsFloatType = FUNK;
        for (int i = 0; i < nKv; i += 2) {
            int key = buffer[i];
            int value = buffer[i + 1];
            if (key == VERSION) spec.version = value;
            else if (key == ARCH_TYPE) spec.archType = (TransformerArchType)value;
            else if (key == DIM) spec.dim = value;
            else if (key == HIDDEN_DIM) spec.hiddenDim = value;
            else if (key == N_LAYERS) spec.nLayers = value;
            else if (key == N_HEADS) spec.nHeads = value;
            else if (key == N_KV_HEADS) spec.nKvHeads = value;
            else if (key == N_EXPERTS) spec.nExperts = value;
            else if (key == N_ACTIVE_EXPERTS) spec.nActiveExperts = value;
            else if (key == VOCAB_SIZE) spec.vocabSize = value;
            else if (key == SEQ_LEN) spec.seqLen = value;
            else if (key == HIDDEN_ACT) spec.hiddenAct = (TransformerHiddenAct)value;
            else if (key == ROPE_THETA) spec.ropeTheta = (float)value;
            else if (key == WEIGHTS_FLOAT_TYPE) weightsFloatType = (FloatType)value;
            else if (key == ROPE_SCALING_FACTOR) spec.ropeScalingFactor = (float)value;
            else if (key == ROPE_SCALING_LOW_FREQ_FACTOR) spec.ropeScalingLowFreqFactor = (float)value;
            else if (key == ROPE_SCALING_HIGH_FREQ_FACTORY) spec.ropeScalingHighFreqFactory = (float)value;
            else if (key == ROPE_SCALING_ORIG_MAX_SEQ_LEN) spec.ropeScalingOrigMaxSeqLen = value;
            else if (key == ROPE_TYPE) spec.ropeType = (TransformerRopeType)value;
            else {
                throw std::runtime_error("Unsupported header key");
            }
        }
    } else {
        throw std::runtime_error("Unsupported model file");
    }

    if (weightsFloatType == FUNK)
        throw std::runtime_error("Not specified weights float type");

    if (spec.ropeType == ROPE_UNKNOWN) {
        if (spec.archType == LLAMA) {
            spec.ropeType = ROPE_LLAMA;
        } else if (spec.archType == GROK1 || spec.archType == MIXTRAL) {
            spec.ropeType = ROPE_FALCON;
        } else {
            throw std::runtime_error("Cannot resolve rope type from architecture");
        }
    }

    spec.origSeqLen = spec.seqLen;
    if (maxSeqLen > 0 && spec.seqLen > maxSeqLen) {
        spec.seqLen = maxSeqLen;
    }
    spec.headSize = spec.dim / spec.nHeads;
    spec.kvDim = (spec.dim * spec.nKvHeads) / spec.nHeads;
    spec.weightsFloatType = weightsFloatType;
    spec.bufferFloatType = bufferFloatType;
    spec.nSlices = nSlices;

    if (spec.nSlices > spec.nKvHeads) {
        // TODO: https://github.com/b4rtaz/distributed-llama/issues/70
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model.");
    }
    if (spec.archType == LLAMA) {
        printf("💡 arch: llama\n");
    } else if (spec.archType == GROK1) {
        printf("💡 arch: grok1\n");
    } else if (spec.archType == MIXTRAL) {
        printf("💡 arch: mixtral\n");
    } else {
        throw std::runtime_error("Unsupported architecture");
    }
    if (spec.hiddenAct == GELU) {
        printf("💡 hiddenAct: gelu\n");
    } else if (spec.hiddenAct == SILU) {
        printf("💡 hiddenAct: silu\n");
    } else {
        throw std::runtime_error("Unsupported hidden activation");
    }
    printf("💡 dim: %d\n", spec.dim);
    printf("💡 hiddenDim: %d\n", spec.hiddenDim);
    printf("💡 nLayers: %d\n", spec.nLayers);
    printf("💡 nHeads: %d\n", spec.nHeads);
    printf("💡 nKvHeads: %d\n", spec.nKvHeads);
    if (spec.nExperts > 0) {
        printf("💡 nExperts: %d\n", spec.nExperts);
        printf("💡 nActiveExperts: %d\n", spec.nActiveExperts);
    }
    printf("💡 vocabSize: %d\n", spec.vocabSize);
    if (spec.seqLen != spec.origSeqLen) {
        printf("💡 origSeqLen: %d\n", spec.origSeqLen);
    }
    printf("💡 seqLen: %d\n", spec.seqLen);
    printf("💡 nSlices: %d\n", spec.nSlices);
    printf("💡 ropeTheta: %.1f\n", spec.ropeTheta);

    spec.fileSize = (size_t)seekToEnd(fd);
    fclose(fd);
    return spec;
}

TransformerBuffer::TransformerBuffer(TransformerSpec* spec) {
    nSlices = spec->nSlices;
    buffers = new void*[TB_LENGTH];
    bufferBytes = new size_t[TB_LENGTH];

    bufferBytes[TB_UNIT_XB] = spec->dim * sizeof(float);
    bufferBytes[TB_UNIT_XB_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, 1);
    bufferBytes[TB_SLICED_XB2] = spec->dim * sizeof(float);
    bufferBytes[TB_SLICED_XB2_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, 1);
    bufferBytes[TB_SLICED_XBV] = spec->dim * spec->nSlices * sizeof(float);
    bufferBytes[TB_SLICED_XBV_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, spec->nSlices);

    int nHb = (spec->nActiveExperts > 0)
        ? spec->hiddenDim * spec->nActiveExperts
        : spec->hiddenDim;
    bufferBytes[TB_SLICED_HB] = nHb * sizeof(float);
    bufferBytes[TB_SLICED_HB_QUANTIZED] = getBatchBytes(spec->bufferFloatType, nHb, 1);

    if (spec->nActiveExperts > 0) {
        bufferBytes[TB_UNIT_MOE_INDEXES] = spec->nActiveExperts * sizeof(uint8_t);
        bufferBytes[TB_UNIT_MOE_WEIGHTS] = spec->nActiveExperts * sizeof(float);

        buffers[TB_UNIT_MOE_INDEXES] = newBuffer(bufferBytes[TB_UNIT_MOE_INDEXES]);
        buffers[TB_UNIT_MOE_WEIGHTS] = newBuffer(bufferBytes[TB_UNIT_MOE_WEIGHTS]);
    } else {
        bufferBytes[TB_UNIT_MOE_INDEXES] = 0;
        bufferBytes[TB_UNIT_MOE_WEIGHTS] = 0;
    }

    for (int i = 0; i < TB_LENGTH - TB_NO_PAIRS; i += 2) {
        int bytes = bufferBytes[i];
        buffers[i] = newBuffer(bufferBytes[i]);
        if (spec->bufferFloatType == F32) {
            buffers[i + 1] = buffers[i];
        } else {
            buffers[i + 1] = newBuffer(bufferBytes[i + 1]);
        }
    }
}

TransformerBuffer::~TransformerBuffer() {
    if (bufferBytes[TB_UNIT_MOE_INDEXES] > 0 && bufferBytes[TB_UNIT_MOE_WEIGHTS] > 0) {
        freeBuffer(buffers[TB_UNIT_MOE_INDEXES]);
        freeBuffer(buffers[TB_UNIT_MOE_WEIGHTS]);
    }

    for (int i = 0; i < TB_LENGTH - TB_NO_PAIRS; i += 2) {
        if (bufferBytes[i] > 0) {
            if (buffers[i] != buffers[i + 1]) {
                freeBuffer(buffers[i + 1]);
            }
            freeBuffer(buffers[i]);
        }
    }
    delete[] bufferBytes;
    delete[] buffers;
}

void* TransformerBuffer::getUnit(uint8_t bufferIndex) {
    return buffers[bufferIndex];
}

size_t TransformerBuffer::getUnitBytes(uint8_t bufferIndex) {
    return bufferBytes[bufferIndex];
}

void* TransformerBuffer::getSliced(uint8_t bufferIndex, slice_index_t sliceIndex) {
    size_t sliceBytes = getSlicedBytes(bufferIndex);
    return ((char*)buffers[bufferIndex]) + sliceBytes * sliceIndex;
}

size_t TransformerBuffer::getSlicedBytes(uint8_t bufferIndex) {
    return bufferBytes[bufferIndex] / nSlices;
}

Transformer::Transformer(TransformerSpec* spec, TransformerConfig* config, slice_index_t sliceIndex) {
    this->spec = spec;
    this->sliceIndex = sliceIndex;

    buffer = new TransformerBuffer(spec);
    blocks = new TransformerBlock*[spec->nLayers];
    for (int i = 0; i < spec->nLayers; i++) {
        blocks[i] = new TransformerBlock(spec, config, sliceIndex);
    }

    if (IS_ROOT_SLICE(sliceIndex)) {
        tokenEmbeddingTableBytes = spec->vocabSize * spec->dim * sizeof(float);
        rmsFinalBytes = spec->dim * sizeof(float);

#if ALLOC_MEMORY
        tokenEmbeddingTable = (float*)newBuffer(tokenEmbeddingTableBytes);
        rmsFinal = (float*)newBuffer(rmsFinalBytes);
#endif
        wclsMm = new MatmulCommand(spec->dim, spec->vocabSize, F32, spec->weightsFloatType);

        x = (float*)newBuffer(spec->dim * sizeof(float));
        logits = (float*)newBuffer(spec->vocabSize * sizeof(float));
    }

    ropeSlice = new RopeSlice(spec->dim, spec->kvDim, spec->nKvHeads, spec->nSlices, spec->seqLen, spec->headSize, spec->ropeTheta, sliceIndex);
    if (spec->ropeType == ROPE_FALCON) {
        rope = new FalconRopeCommand(ropeSlice);
    } else if (spec->ropeType == ROPE_LLAMA) {
        rope = new LlamaRopeCommand(ropeSlice);
    } else if (spec->ropeType == ROPE_LLAMA3_1) {
        rope = new Llama3_1RopeCommand(ropeSlice, spec->ropeScalingFactor, spec->ropeScalingLowFreqFactor, spec->ropeScalingHighFreqFactory, spec->ropeScalingOrigMaxSeqLen);
    } else {
        throw std::runtime_error("Unsupported rope type");
    }

    TransformerBlock* b = blocks[0];
    assert(b->q0Slice->d0 == ropeSlice->qDim0);
    assert(b->q0Slice->dOffset(sliceIndex) == ropeSlice->qDimStart);
    assert(b->k0Slice->d0 == ropeSlice->kvDim0);
    assert(b->k0Slice->dOffset(sliceIndex) == ropeSlice->kvDimStart);
    assert(b->kvCacheSlice->kvDim0 == ropeSlice->kvDim0);
}

Transformer::~Transformer() {
    delete buffer;
    for (int i = 0; i < spec->nLayers; i++) {
        delete blocks[i];
    }
    delete[] blocks;

    if (IS_ROOT_SLICE(sliceIndex)) {
#if ALLOC_MEMORY
        freeBuffer(tokenEmbeddingTable);
        freeBuffer(rmsFinal);
#endif
        delete wclsMm;

        freeBuffer(x);
        freeBuffer(logits);
    }

    delete ropeSlice;
    delete rope;
}

TransformerBlock::TransformerBlock(TransformerSpec* spec, TransformerConfig* config, slice_index_t sliceIndex) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;
    this->config = config;

    if (IS_ROOT_SLICE(sliceIndex)) {
        rmsAttBytes = spec->dim * sizeof(float);
        rmsFfnBytes = spec->dim * sizeof(float);
        rmsMoeBytes = spec->dim * sizeof(float);
        rmsFfn2Bytes = spec->dim * sizeof(float);

#if ALLOC_MEMORY
        rmsAtt = (float*)newBuffer(rmsAttBytes);
        rmsFfn = (float*)newBuffer(rmsFfnBytes);
        if (spec->archType == GROK1) {
            rmsMoe = (float*)newBuffer(rmsMoeBytes);
            rmsFfn2 = (float*)newBuffer(rmsFfn2Bytes);
        }
#endif
    }

    kvCacheSlice = new KvCacheSlice(spec->kvDim, spec->seqLen, spec->nSlices);
    if (config->useDiscForKvCache) {
        keyCache = (float*)newMmapFileBuffer(sliceIndex, kvCacheSlice->keyCacheSize);
        valueCache = (float*)newMmapFileBuffer(sliceIndex, kvCacheSlice->valueCacheSize);
    } else {
        keyCache = (float*)newBuffer(kvCacheSlice->keyCacheSize);
        valueCache = (float*)newBuffer(kvCacheSlice->valueCacheSize);
    }

    multiHeadAttSlice = new MultiHeadAttSlice(spec->nHeads, spec->seqLen, spec->nSlices, sliceIndex);
    att = (float*)newBuffer(multiHeadAttSlice->attSize);

    q0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->dim);
    k0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->kvDim);
    v0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->kvDim);
    wo0Slice = new ColMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->dim);

    q0mm = new MatmulCommand(q0Slice->n, q0Slice->d0, spec->bufferFloatType, spec->weightsFloatType);
    k0mm = new MatmulCommand(k0Slice->n, k0Slice->d0, spec->bufferFloatType, spec->weightsFloatType);
    v0mm = new MatmulCommand(v0Slice->n, v0Slice->d0, spec->bufferFloatType, spec->weightsFloatType);
    wo0mm = new MatmulCommand(wo0Slice->n0, wo0Slice->d, spec->bufferFloatType, spec->weightsFloatType);

    qo0 = (float*)newBuffer(q0Slice->d0 * sizeof(float));

    if (spec->nExperts > 0) {
        moeUpAndGate0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim);
        moeDown0Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->hiddenDim, spec->dim);

        moeRouterProbs = (float*)newBuffer(spec->nExperts * sizeof(float));

        moeUpMm = new MatmulCommand*[spec->nExperts];
        moeGateMm = new MatmulCommand*[spec->nExperts];
        moeDownMm = new MatmulCommand*[spec->nExperts];
        moeRouterMm = new MatmulCommand(spec->dim, spec->nExperts, F32, spec->weightsFloatType);

        for (int e = 0; e < spec->nExperts; e++) {
            moeUpMm[e] = new MatmulCommand(moeUpAndGate0Slice->n, moeUpAndGate0Slice->d0, spec->bufferFloatType, spec->weightsFloatType);
            moeGateMm[e] = new MatmulCommand(moeUpAndGate0Slice->n, moeUpAndGate0Slice->d0, spec->bufferFloatType, spec->weightsFloatType);
            moeDownMm[e] = new MatmulCommand(moeDown0Slice->n, moeDown0Slice->d0, spec->bufferFloatType, spec->weightsFloatType);
        }

        expertGate = (float*)newBuffer(moeUpAndGate0Slice->d0 * spec->nExperts * sizeof(float));
        expertDown = (float*)newBuffer(moeDown0Slice->d0 * (spec->nExperts - 1) * sizeof(float));
    } else {
        w10Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim);
        w20Slice = new ColMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->hiddenDim, spec->dim);
        w30Slice = new RowMatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim);

        w10mm = new MatmulCommand(w10Slice->n, w10Slice->d0, spec->bufferFloatType, spec->weightsFloatType);
        w20mm = new MatmulCommand(w20Slice->n0, w20Slice->d, spec->bufferFloatType, spec->weightsFloatType);
        w30mm = new MatmulCommand(w30Slice->n, w30Slice->d0, spec->bufferFloatType, spec->weightsFloatType);

        hb20 = (float*)newBuffer(w30Slice->d0 * sizeof(float));
    }
}

TransformerBlock::~TransformerBlock() {
#if ALLOC_MEMORY
    if (IS_ROOT_SLICE(sliceIndex)) {
        freeBuffer(rmsAtt);
        freeBuffer(rmsFfn);
        if (spec->archType == GROK1) {
            freeBuffer(rmsMoe);
            freeBuffer(rmsFfn2);
        }
    }
#endif

    delete kvCacheSlice;
    if (config->useDiscForKvCache) {
        freeMmapFileBuffer(keyCache);
        freeMmapFileBuffer(valueCache);
    } else {
        freeBuffer(keyCache);
        freeBuffer(valueCache);
    }
    delete multiHeadAttSlice;
    freeBuffer(att);

    delete q0Slice;
    delete k0Slice;
    delete v0Slice;
    delete wo0Slice;

    freeBuffer(qo0);

    delete q0mm;
    delete k0mm;
    delete v0mm;
    delete wo0mm;

    if (spec->nExperts > 0) {
        delete moeUpAndGate0Slice;
        delete moeDown0Slice;

        for (int e = 0; e < spec->nExperts; e++) {
            delete moeUpMm[e];
            delete moeGateMm[e];
            delete moeDownMm[e];
        }

        delete[] moeUpMm;
        delete[] moeGateMm;
        delete[] moeDownMm;
        freeBuffer(moeRouterProbs);

        freeBuffer(expertGate);
        freeBuffer(expertDown);
    } else {
        delete w10Slice;
        delete w20Slice;
        delete w30Slice;

        delete w10mm;
        delete w20mm;
        delete w30mm;

        freeBuffer(hb20);
    }
}

static size_t loadSlicedMatmulWeights(const uint8_t nSlices, MatmulSlice* slice, char* source, MatmulCommand* mm, SocketPool* socketPool) {
#if ALLOC_MEMORY
    char* buffer = (char*)newBuffer(slice->sliceBytes);
    size_t loadedBytes = 0;
    for (uint8_t s = 0; s < nSlices; s++) {
        slice_index_t sliceIndex = (s + 1) % nSlices;
        loadedBytes += slice->splitWeights(sliceIndex, source, buffer);
        if (sliceIndex > 0) {
            unsigned int socketIndex = sliceIndex - 1;
            socketPool->write(socketIndex, buffer, slice->sliceBytes);
        } else {
            mm->loadWeights(buffer);
        }
    }
    freeBuffer(buffer);
    return loadedBytes;
#else
    return mm->loadWeights(source);
#endif
}

static size_t loadRootWeights(char** target, char* source, size_t bytes) {
#if ALLOC_MEMORY
    memcpy(*target, source, bytes);
#else
    *target = source;
#endif
    return bytes;
}

static size_t readSlicedMatmulWeights(MatmulSlice* slice, char* weights0, Socket* socket) {
    socket->read(weights0, slice->sliceBytes);
    return slice->sliceBytes;
}

Transformer Transformer::loadRootFromFile(const char* path, TransformerSpec* spec, TransformerConfig* config, SocketPool* socketPool) {
    MmapFile file;
    openMmapFile(&file, path, spec->fileSize);

    char* weights = ((char*)file.data) + spec->headerSize;
    Transformer transformer = Transformer::loadRoot((char*)weights, spec, config, socketPool);

#if ALLOC_MEMORY
    closeMmapFile(&file);
#endif
    return transformer;
}

Transformer Transformer::loadRoot(char* data, TransformerSpec* spec, TransformerConfig* config, SocketPool* socketPool) {
    assert(socketPool->nSockets == spec->nSlices - 1);

    const slice_index_t sliceIndex = 0; // Root slice
    Transformer transformer(spec, config, sliceIndex);

    if (spec->nSlices > 1) {
        for (slice_index_t sliceIndex = 1; sliceIndex < spec->nSlices; sliceIndex++) {
            unsigned int socketIndex = sliceIndex - 1;
            socketPool->write(socketIndex, (char*)&sliceIndex, sizeof(uint8_t));
            socketPool->write(socketIndex, (char*)spec, sizeof(TransformerSpec));
        }
    }

    char* w = data;

    w += loadRootWeights((char**)&transformer.tokenEmbeddingTable, w, transformer.tokenEmbeddingTableBytes);

    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];
        w += loadSlicedMatmulWeights(spec->nSlices, block->q0Slice, w, block->q0mm, socketPool);
        w += loadSlicedMatmulWeights(spec->nSlices, block->k0Slice, w, block->k0mm, socketPool);
        w += loadSlicedMatmulWeights(spec->nSlices, block->v0Slice, w, block->v0mm, socketPool);
        w += loadSlicedMatmulWeights(spec->nSlices, block->wo0Slice, w, block->wo0mm, socketPool);

        if (spec->nExperts > 0) {
            w += block->moeRouterMm->loadWeights(w);

            for (int e = 0; e < spec->nExperts; e++) {
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeUpAndGate0Slice, w, block->moeUpMm[e], socketPool);
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeUpAndGate0Slice, w, block->moeGateMm[e], socketPool);
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeDown0Slice, w, block->moeDownMm[e], socketPool);
            }
        } else {
            w += loadSlicedMatmulWeights(spec->nSlices, block->w10Slice, w, block->w10mm, socketPool);
            w += loadSlicedMatmulWeights(spec->nSlices, block->w20Slice, w, block->w20mm, socketPool);
            w += loadSlicedMatmulWeights(spec->nSlices, block->w30Slice, w, block->w30mm, socketPool);
        }

        w += loadRootWeights((char**)&block->rmsAtt, w, block->rmsAttBytes);
        w += loadRootWeights((char**)&block->rmsFfn, w, block->rmsFfnBytes);

        if (spec->archType == GROK1) {
            w += loadRootWeights((char**)&block->rmsMoe, w, block->rmsMoeBytes);
            w += loadRootWeights((char**)&block->rmsFfn2, w, block->rmsFfn2Bytes);
        }
    }

    w += loadRootWeights((char**)&transformer.rmsFinal, w, transformer.rmsFinalBytes);
    w += transformer.wclsMm->loadWeights(w);

    long missedBytes = (long)(w - data) - spec->fileSize + spec->headerSize;
    if (missedBytes != 0) {
        printf("The model file is missing %ld bytes\n", missedBytes);
        exit(EXIT_FAILURE);
    }

    printf("⏩ Loaded %ld kB\n", (long)(w - data) / 1024);
    return transformer;
}

Transformer Transformer::loadSlice(TransformerSpec* spec, TransformerConfig* config, Socket* socket) {
    slice_index_t sliceIndex;
    socket->read((char*)&sliceIndex, sizeof(uint8_t));
    socket->read((char*)spec, sizeof(TransformerSpec));

    printf("💡 sliceIndex: %d\n", sliceIndex);
    printf("💡 nSlices: %d\n", spec->nSlices);

    assert(sliceIndex >= 1);
    Transformer transformer(spec, config, sliceIndex);

    size_t bufferSize = 0;
    // TODO: this is ugly
    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];
        if (block->k0Slice->sliceBytes > bufferSize) bufferSize = block->k0Slice->sliceBytes;
        if (block->q0Slice->sliceBytes > bufferSize) bufferSize = block->q0Slice->sliceBytes;
        if (block->wo0Slice->sliceBytes > bufferSize) bufferSize = block->wo0Slice->sliceBytes;
        if (spec->nExperts > 0) {
            if (block->moeUpAndGate0Slice[0].sliceBytes > bufferSize) bufferSize = block->moeUpAndGate0Slice[0].sliceBytes;
            if (block->moeDown0Slice[0].sliceBytes > bufferSize) bufferSize = block->moeDown0Slice[0].sliceBytes;
        } else {
            if (block->w10Slice->sliceBytes > bufferSize) bufferSize = block->w10Slice->sliceBytes;
            if (block->w20Slice->sliceBytes > bufferSize) bufferSize = block->w20Slice->sliceBytes;
            if (block->w30Slice->sliceBytes > bufferSize) bufferSize = block->w30Slice->sliceBytes;
        }
    }

    char* buffer = new char[bufferSize];

    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];
        size_t blockBytes = 0;
        long t0 = timeMs();

        socket->read(buffer, block->q0Slice->sliceBytes);
        blockBytes += block->q0mm->loadWeights(buffer);

        socket->read(buffer, block->k0Slice->sliceBytes);
        blockBytes += block->k0mm->loadWeights(buffer);

        socket->read(buffer, block->v0Slice->sliceBytes);
        blockBytes += block->v0mm->loadWeights(buffer);

        socket->read(buffer, block->wo0Slice->sliceBytes);
        blockBytes += block->wo0mm->loadWeights(buffer);

        if (spec->nExperts > 0) {
            for (int e = 0; e < spec->nExperts; e++) {
                socket->read(buffer, block->moeUpAndGate0Slice->sliceBytes);
                blockBytes += block->moeUpMm[e]->loadWeights(buffer);

                socket->read(buffer, block->moeUpAndGate0Slice->sliceBytes);
                blockBytes += block->moeGateMm[e]->loadWeights(buffer);

                socket->read(buffer, block->moeDown0Slice->sliceBytes);
                blockBytes += block->moeDownMm[e]->loadWeights(buffer);
            }
        } else {
            socket->read(buffer, block->w10Slice->sliceBytes);
            blockBytes += block->w10mm->loadWeights(buffer);

            socket->read(buffer, block->w20Slice->sliceBytes);
            blockBytes += block->w20mm->loadWeights(buffer);

            socket->read(buffer, block->w30Slice->sliceBytes);
            blockBytes += block->w30mm->loadWeights(buffer);
        }

        float kbs = blockBytes / (float)(timeMs() - t0);
        printf("⏩ Received %ld kB for block %d (%.0f kB/s)\n", blockBytes / 1024, i, kbs);
    }

    delete[] buffer;
    return transformer;
}
