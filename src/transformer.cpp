#include <cstdio>
#include <cmath>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <cassert>
#include "utils.hpp"
#include "socket.hpp"
#include "transformer.hpp"
#include <unistd.h>
#include <stdexcept>

#define ALLOC_WEIGHTS true
#define IS_ROOT_SLICE(sliceIndex) (sliceIndex == 0)

MatmulSlice::MatmulSlice(FloatType type, int nSlices, int n, int d) {
    assert(d % nSlices == 0);

    this->type = type;
    this->nSlices = nSlices;
    this->d0 = d / nSlices;
    this->n = n;
    this->bytes = getBatchBytes(type, this->n, d);
    this->sliceBytes = getBatchBytes(type, this->n, this->d0);
}

size_t MatmulSlice::splitWeights(uint8_t sliceIndex, char* weights, char* weights0) {
    int numbersPerBatch = getNumbersPerBatch(this->type);
    int batchBytes = getBatchBytes(this->type, numbersPerBatch, 1);

    int n = this->n / numbersPerBatch;
    size_t offset = this->d0 * sliceIndex * n * batchBytes;
    size_t copiedBytes = 0;

    for (int d = 0; d < this->d0; d++) {
        for (int j = 0; j < n; j++) {
            long o = (d * n + j) * batchBytes;

            memcpy(weights0 + o, weights + offset + o, batchBytes);
            copiedBytes += batchBytes;
        }
    }
    return copiedBytes;
}

void initRope(float* cache, TransformerSpec* spec) {
    for (pos_t pos = 0; pos < spec->seqLen; pos++) {
        for (int i = 0; i < spec->dim; i += 2) {
            int head_dim = i % spec->headSize;
            float freq = 1.0f / powf(spec->ropeTheta, head_dim / (float)spec->headSize);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            cache[pos * spec->dim + i] = fcr;
            cache[pos * spec->dim + i + 1] = fci;
        }
    }
}

void rope(float* cache, float* q, float* k, TransformerSpec* spec, pos_t pos, unsigned int nThreads, unsigned int threadIndex) {
    int halfDim = spec->dim / 2;
    int slice = halfDim / nThreads;
    int iStart = threadIndex * slice;
    int iEnd = ((nThreads - 1 == threadIndex) ? halfDim : (iStart + slice)) * 2;
    iStart *= 2;

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = iStart; i < iEnd; i += 2) {
        float fcr = cache[pos * spec->dim + i];
        float fci = cache[pos * spec->dim + i + 1];
        int rotn = i < spec->kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
        for (int _v = 0; _v < rotn; _v++) {
            float* vec = _v == 0 ? q : k; // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i+1];
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

TransformerSpec Transformer::loadSpecFromFile(const char* path, const unsigned int nSlices, FloatType weightsFloatType, FloatType bufferFloatType) {
    TransformerSpec spec;
    memset(&spec, 0, sizeof(TransformerSpec));
    spec.hiddenAct = GELU;
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
            else {
                throw std::runtime_error("Unsupported header key");
            }
        }
    } else {
        throw std::runtime_error("Unsupported model file");
    }

    spec.headSize = spec.dim / spec.nHeads;
    spec.kvDim = (spec.dim * spec.nKvHeads) / spec.nHeads;
    spec.weightsFloatType = weightsFloatType;
    spec.bufferFloatType = bufferFloatType;
    spec.nSlices = nSlices;

    if (spec.archType == LLAMA2) {
        printf("💡 arch: llama2\n");
    } else if (spec.archType == GROK1) {
        printf("💡 arch: grok1\n");
    } else if (spec.archType == MIXTRAL) {
        printf("💡 arch: mixtral\n");
    } else {
        throw std::runtime_error("Unsupported architecture");
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
    printf("💡 seqLen: %d\n", spec.seqLen);
    printf("💡 nSlices: %d\n", spec.nSlices);
    printf("💡 ropeTheta: %.1f\n", spec.ropeTheta);

    fseek(fd, 0, SEEK_END);
    size_t fileSize = ftell(fd);
    fclose(fd);

    spec.fileSize = fileSize;
    return spec;
}

TransformerBuffer::TransformerBuffer(TransformerSpec* spec) {
    nSlices = spec->nSlices;
    buffers = new char*[TB_LENGTH];
    bufferBytes = new size_t[TB_LENGTH];

    bufferBytes[TB_UNIT_XB] = spec->dim * sizeof(float);
    bufferBytes[TB_UNIT_XB_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, 1);
    bufferBytes[TB_SLICED_XB2] = spec->dim * sizeof(float);
    bufferBytes[TB_SLICED_XB2_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, 1);
    bufferBytes[TB_SLICED_Q] = spec->dim * sizeof(float);
    bufferBytes[TB_SLICED_Q_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->dim, 1);
    bufferBytes[TB_SLICED_K] = spec->kvDim * sizeof(float);
    bufferBytes[TB_SLICED_K_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->kvDim, 1);
    bufferBytes[TB_SLICED_V] = spec->kvDim * sizeof(float);
    bufferBytes[TB_SLICED_V_QUANTIZED] = getBatchBytes(spec->bufferFloatType, spec->kvDim, 1);

    int nHb = (spec->nActiveExperts > 0)
        ? spec->hiddenDim * spec->nActiveExperts
        : spec->hiddenDim;
    bufferBytes[TB_SLICED_HB] = nHb * sizeof(float);
    bufferBytes[TB_SLICED_HB_QUANTIZED] = getBatchBytes(spec->bufferFloatType, nHb, 1);

    if (spec->nActiveExperts > 0) {
        bufferBytes[TB_UNIT_MOE_INDEXES] = spec->nActiveExperts * sizeof(uint8_t);
        bufferBytes[TB_UNIT_MOE_WEIGHTS] = spec->nActiveExperts * sizeof(float);

        buffers[TB_UNIT_MOE_INDEXES] = NEW_BUFFER(bufferBytes[TB_UNIT_MOE_INDEXES]);
        buffers[TB_UNIT_MOE_WEIGHTS] = NEW_BUFFER(bufferBytes[TB_UNIT_MOE_WEIGHTS]);
    } else {
        bufferBytes[TB_UNIT_MOE_INDEXES] = 0;
        bufferBytes[TB_UNIT_MOE_WEIGHTS] = 0;
    }

    for (int i = 0; i < TB_LENGTH - TB_NO_PAIRS; i += 2) {
        int bytes = bufferBytes[i];
        buffers[i] = NEW_BUFFER(bufferBytes[i]);
        if (spec->bufferFloatType == F32) {
            buffers[i + 1] = buffers[i];
        } else {
            buffers[i + 1] = NEW_BUFFER(bufferBytes[i + 1]);
        }
    }
}

TransformerBuffer::~TransformerBuffer() {
    if (bufferBytes[TB_UNIT_MOE_INDEXES] > 0 && bufferBytes[TB_UNIT_MOE_WEIGHTS] > 0) {
        FREE_BUFFER(buffers[TB_UNIT_MOE_INDEXES]);
        FREE_BUFFER(buffers[TB_UNIT_MOE_WEIGHTS]);
    }

    for (int i = 0; i < TB_LENGTH - TB_NO_PAIRS; i += 2) {
        if (bufferBytes[i] > 0) {
            if (buffers[i] != buffers[i + 1]) {
                FREE_BUFFER(buffers[i + 1]);
            }
            FREE_BUFFER(buffers[i]);
        }
    }
    delete[] bufferBytes;
    delete[] buffers;
}

char* TransformerBuffer::getUnit(uint8_t bufferIndex) {
    return buffers[bufferIndex];
}

size_t TransformerBuffer::getUnitBytes(uint8_t bufferIndex) {
    return bufferBytes[bufferIndex];
}

char* TransformerBuffer::getSliced(uint8_t bufferIndex, uint8_t sliceIndex) {
    size_t sliceBytes = getSlicedBytes(bufferIndex);
    return buffers[bufferIndex] + sliceBytes * sliceIndex;
}

size_t TransformerBuffer::getSlicedBytes(uint8_t bufferIndex) {
    return bufferBytes[bufferIndex] / nSlices;
}

Transformer::Transformer(TransformerSpec* spec, uint8_t sliceIndex) {
    this->spec = spec;
    this->sliceIndex = sliceIndex;

    buffer = new TransformerBuffer(spec);
    blocks = new TransformerBlock*[spec->nLayers];
    for (int i = 0; i < spec->nLayers; i++) {
        blocks[i] = new TransformerBlock(spec, sliceIndex);
    }

    if (IS_ROOT_SLICE(sliceIndex)) {
        tokenEmbeddingTableBytes = spec->vocabSize * spec->dim * sizeof(float);
        rmsFinalBytes = spec->dim * sizeof(float);
        wclsBytes = getBatchBytes(spec->weightsFloatType, spec->vocabSize, spec->dim);
#if ALLOC_WEIGHTS
        tokenEmbeddingTable = NEW_BUFFER(tokenEmbeddingTableBytes);
        rmsFinal = NEW_BUFFER(rmsFinalBytes);
        wcls = NEW_BUFFER(wclsBytes);
#endif
        x = (float*)NEW_BUFFER(spec->dim * sizeof(float));
        logits = (float*)NEW_BUFFER(spec->vocabSize * sizeof(float));

        // TODO: cache should be for all architectures
        if (spec->archType == LLAMA2 || spec->archType == MIXTRAL) {
            ropeCache = (float*)NEW_BUFFER(spec->seqLen * spec->dim * sizeof(float));
            initRope(ropeCache, spec);
        } else {
            ropeCache = NULL;
        }
    }
}

Transformer::~Transformer() {
    delete buffer;
    for (int i = 0; i < spec->nLayers; i++) {
        delete blocks[i];
    }
    delete[] blocks;

    if (IS_ROOT_SLICE(sliceIndex)) {
#if ALLOC_WEIGHTS
        FREE_BUFFER(tokenEmbeddingTable);
        FREE_BUFFER(rmsFinal);
        FREE_BUFFER(wcls);
#endif
        FREE_BUFFER(x);
        FREE_BUFFER(logits);

        if (ropeCache != NULL) {
            FREE_BUFFER(ropeCache);
        }
    }
}

TransformerBlock::TransformerBlock(TransformerSpec* spec, uint8_t sliceIndex) {
    this->sliceIndex = sliceIndex;
    this->spec = spec;

    if (IS_ROOT_SLICE(sliceIndex)) {
        rmsAttBytes = spec->dim * sizeof(float);
        rmsFfnBytes = spec->dim * sizeof(float);
        rmsMoeBytes = spec->dim * sizeof(float);
        rmsFfn2Bytes = spec->dim * sizeof(float);
#if ALLOC_WEIGHTS
        rmsAtt = (float*)NEW_BUFFER(rmsAttBytes);
        rmsFfn = (float*)NEW_BUFFER(rmsFfnBytes);
        if (spec->archType == GROK1) {
            rmsMoe = (float*)NEW_BUFFER(rmsMoeBytes);
            rmsFfn2 = (float*)NEW_BUFFER(rmsFfn2Bytes);
        }
#endif

        keyCache = (float*)NEW_BUFFER(spec->seqLen * spec->kvDim * sizeof(float));
        valueCache = (float*)NEW_BUFFER(spec->seqLen * spec->kvDim * sizeof(float));
        att = (float*)NEW_BUFFER(spec->nHeads * spec->seqLen * sizeof(float));
    }

    q0Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->dim);
    k0Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->kvDim);
    v0Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->kvDim);
    wo0Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->dim);

#if ALLOC_WEIGHTS
    q0 = NEW_BUFFER(q0Slice->sliceBytes);
    k0 = NEW_BUFFER(k0Slice->sliceBytes);
    v0 = NEW_BUFFER(v0Slice->sliceBytes);
    wo0 = NEW_BUFFER(wo0Slice->sliceBytes);
#endif

    if (spec->nExperts > 0) {
        moeUpAndGate0Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim);
        moeDown0Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->hiddenDim, spec->dim);

        moeRouterBytes = getBatchBytes(spec->weightsFloatType, spec->dim, spec->nExperts);
        moeRouterProbs = (float*)NEW_BUFFER(spec->nExperts * sizeof(float));

        moeUp = new char*[spec->nExperts];
        moeGate = new char*[spec->nExperts];
        moeDown = new char*[spec->nExperts];

#if ALLOC_WEIGHTS
        moeRouter = NEW_BUFFER(moeRouterBytes);

        for (int e = 0; e < spec->nExperts; e++) {
            moeUp[e] = NEW_BUFFER(moeUpAndGate0Slice->sliceBytes);
            moeGate[e] = NEW_BUFFER(moeUpAndGate0Slice->sliceBytes);
            moeDown[e] = NEW_BUFFER(moeDown0Slice->sliceBytes);
        }
#endif
        expertGate = (float*)NEW_BUFFER(moeUpAndGate0Slice->d0 * spec->nExperts * sizeof(float));
        expertDown = (float*)NEW_BUFFER(moeDown0Slice->d0 * (spec->nExperts - 1) * sizeof(float));
    } else {
        w10Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim);
        w20Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->hiddenDim, spec->dim);
        w30Slice = new MatmulSlice(spec->weightsFloatType, spec->nSlices, spec->dim, spec->hiddenDim);

#if ALLOC_WEIGHTS
        w10 = NEW_BUFFER(w10Slice->sliceBytes);
        w20 = NEW_BUFFER(w20Slice->sliceBytes);
        w30 = NEW_BUFFER(w30Slice->sliceBytes);
#endif

        hb20 = (float*)NEW_BUFFER(w30Slice->d0 * sizeof(float));
    }
}

TransformerBlock::~TransformerBlock() {
    if (IS_ROOT_SLICE(sliceIndex)) {
#if ALLOC_WEIGHTS
        FREE_BUFFER(rmsAtt);
        FREE_BUFFER(rmsFfn);
        if (spec->archType == GROK1) {
            FREE_BUFFER(rmsMoe);
            FREE_BUFFER(rmsFfn2);
        }
#endif
        FREE_BUFFER(keyCache);
        FREE_BUFFER(valueCache);
        FREE_BUFFER(att);
    }

    delete q0Slice;
    delete k0Slice;
    delete v0Slice;
    delete wo0Slice;

#if ALLOC_WEIGHTS
    FREE_BUFFER(q0);
    FREE_BUFFER(k0);
    FREE_BUFFER(v0);
    FREE_BUFFER(wo0);
#endif

    if (spec->nExperts > 0) {
        delete moeUpAndGate0Slice;
        delete moeDown0Slice;

#if ALLOC_WEIGHTS
        for (int e = 0; e < spec->nExperts; e++) {
            FREE_BUFFER(moeUp[e]);
            FREE_BUFFER(moeGate[e]);
            FREE_BUFFER(moeDown[e]);
        }

        FREE_BUFFER(moeRouter);
#endif
        delete[] moeUp;
        delete[] moeGate;
        delete[] moeDown;
        FREE_BUFFER(moeRouterProbs);

        FREE_BUFFER(expertGate);
        FREE_BUFFER(expertDown);
    } else {
        delete w10Slice;
        delete w20Slice;
        delete w30Slice;

#if ALLOC_WEIGHTS
        FREE_BUFFER(w10);
        FREE_BUFFER(w20);
        FREE_BUFFER(w30);
#endif

        FREE_BUFFER(hb20);
    }
}

static size_t loadSlicedMatmulWeights(uint8_t nSlices, MatmulSlice* slice, char* weights, char** weights0, SocketPool* socketPool) {
#if ALLOC_WEIGHTS
    if (nSlices > 1) {
        char* temp = NEW_BUFFER(slice->bytes);
        memcpy(temp, weights, slice->bytes);

        size_t loadedBytes = 0;
        for (uint8_t s = 0; s < nSlices; s++) {
            uint8_t sliceIndex = (s + 1) % nSlices; // Root slice must be loaded last because we want keep root weights in the memory.
            loadedBytes += slice->splitWeights(sliceIndex, temp, *weights0);
            if (sliceIndex > 0) {
                unsigned int socketIndex = sliceIndex - 1;
                socketPool->write(socketIndex, *weights0, slice->sliceBytes);
            }
        }

        assert(loadedBytes == slice->bytes);
        FREE_BUFFER(temp);
        return loadedBytes;
    } else {
        size_t loadedBytes = slice->splitWeights(0, weights, *weights0);
        assert(loadedBytes == slice->bytes);
        return loadedBytes;
    }
#else
    assert(nSlices == 1);
    *weights0 = weights;
    return slice->bytes;
#endif
}

static size_t loadRootMatmulWeights(char** target, char* source, size_t bytes) {
#if ALLOC_WEIGHTS
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

Transformer Transformer::loadRootFromFile(const char* path, TransformerSpec* spec, SocketPool* socketPool) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        printf("Cannot open file %s\n", path);
        exit(EXIT_FAILURE);
    }
    char* data = (char*)mmap(NULL, spec->fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        printf("Mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    char* weights = data + spec->headerSize;
    Transformer transformer = Transformer::loadRoot(weights, spec, socketPool);
#if ALLOC_WEIGHTS
    munmap(data, spec->fileSize);
    close(fd);
#else
    // TODO: handler should be released in deconstructor
#endif
    return transformer;
}

Transformer Transformer::loadRoot(char* data, TransformerSpec* spec, SocketPool* socketPool) {
    assert(socketPool->nSockets == spec->nSlices - 1);

    const uint8_t sliceIndex = 0; // Root slice
    Transformer transformer(spec, sliceIndex);

    if (spec->nSlices > 1) {
        for (uint8_t sliceIndex = 1; sliceIndex < spec->nSlices; sliceIndex++) {
            unsigned int socketIndex = sliceIndex - 1;
            socketPool->write(socketIndex, (char*)&sliceIndex, sizeof(uint8_t));
            socketPool->write(socketIndex, (char*)spec, sizeof(TransformerSpec));
        }
    }

    char* w = data;

    w += loadRootMatmulWeights(&transformer.tokenEmbeddingTable, w, transformer.tokenEmbeddingTableBytes);

    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];

        w += loadSlicedMatmulWeights(spec->nSlices, block->q0Slice, w, &block->q0, socketPool);
        w += loadSlicedMatmulWeights(spec->nSlices, block->k0Slice, w, &block->k0, socketPool);
        w += loadSlicedMatmulWeights(spec->nSlices, block->v0Slice, w, &block->v0, socketPool);
        w += loadSlicedMatmulWeights(spec->nSlices, block->wo0Slice, w, &block->wo0, socketPool);

        if (spec->nExperts > 0) {
            w += loadRootMatmulWeights(&block->moeRouter, w, block->moeRouterBytes);

            for (int e = 0; e < spec->nExperts; e++) {
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeUpAndGate0Slice, w, &block->moeUp[e], socketPool);
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeUpAndGate0Slice, w, &block->moeGate[e], socketPool);
                w += loadSlicedMatmulWeights(spec->nSlices, block->moeDown0Slice, w, &block->moeDown[e], socketPool);
            }
        } else {
            w += loadSlicedMatmulWeights(spec->nSlices, block->w10Slice, w, &block->w10, socketPool);
            w += loadSlicedMatmulWeights(spec->nSlices, block->w20Slice, w, &block->w20, socketPool);
            w += loadSlicedMatmulWeights(spec->nSlices, block->w30Slice, w, &block->w30, socketPool);
        }

        w += loadRootMatmulWeights((char**)&block->rmsAtt, w, block->rmsAttBytes);
        w += loadRootMatmulWeights((char**)&block->rmsFfn, w, block->rmsFfnBytes);

        if (spec->archType == GROK1) {
            w += loadRootMatmulWeights((char**)&block->rmsMoe, w, block->rmsMoeBytes);
            w += loadRootMatmulWeights((char**)&block->rmsFfn2, w, block->rmsFfn2Bytes);
        }
    }

    w += loadRootMatmulWeights(&transformer.rmsFinal, w, transformer.rmsFinalBytes);
    w += loadRootMatmulWeights(&transformer.wcls, w, transformer.wclsBytes);

    long missedBytes = (long)(w - data) - spec->fileSize + spec->headerSize;
    if (missedBytes != 0) {
        printf("Missed %ld bytes\n", missedBytes);
        exit(EXIT_FAILURE);
    }

    printf("⏩ Loaded %ld bytes\n", (long)(w - data));
    return transformer;
}

Transformer Transformer::loadSlice(TransformerSpec* spec, Socket* socket) {
    uint8_t sliceIndex;
    socket->read((char*)&sliceIndex, sizeof(uint8_t));
    socket->read((char*)spec, sizeof(TransformerSpec));

    printf("💡 sliceIndex: %d\n", sliceIndex);
    printf("💡 nSlices: %d\n", spec->nSlices);

    assert(sliceIndex >= 1);
    Transformer transformer(spec, sliceIndex);

    for (int i = 0; i < spec->nLayers; i++) {
        TransformerBlock* block = transformer.blocks[i];
        size_t blockBytes = 0;
        long t0 = timeMs();
        blockBytes += readSlicedMatmulWeights(block->q0Slice, block->q0, socket);
        blockBytes += readSlicedMatmulWeights(block->k0Slice, block->k0, socket);
        blockBytes += readSlicedMatmulWeights(block->v0Slice, block->v0, socket);
        blockBytes += readSlicedMatmulWeights(block->wo0Slice, block->wo0, socket);

        if (spec->nExperts > 0) {
            for (int e = 0; e < spec->nExperts; e++) {
                blockBytes += readSlicedMatmulWeights(block->moeUpAndGate0Slice, block->moeUp[e], socket);
                blockBytes += readSlicedMatmulWeights(block->moeUpAndGate0Slice, block->moeGate[e], socket);
                blockBytes += readSlicedMatmulWeights(block->moeDown0Slice, block->moeDown[e], socket);
            }
        } else {
            blockBytes += readSlicedMatmulWeights(block->w10Slice, block->w10, socket);
            blockBytes += readSlicedMatmulWeights(block->w20Slice, block->w20, socket);
            blockBytes += readSlicedMatmulWeights(block->w30Slice, block->w30, socket);
        }

        float kbs = blockBytes / (float)(timeMs() - t0);
        printf("⏩ Received %ld bytes for block %d (%.0f kB/s)\n", blockBytes, i, kbs);
    }
    return transformer;
}
