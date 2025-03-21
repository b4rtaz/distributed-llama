#ifndef NN_CONFIG_BUILDER_H
#define NN_CONFIG_BUILDER_H

#include "nn-core.hpp"
#include <cassert>
#include <cstring>

static char *cloneString(const char *str) {
    NnUint len = std::strlen(str);
    char *copy = new char[len + 1];
    std::memcpy(copy, str, len + 1);
    return copy;
}

class NnNetConfigBuilder {
public:
    NnUint nNodes;
    NnUint nBatches;
    std::list<NnPipeConfig> pipes;
    std::list<NnPreSyncConfig> preSyncs;

    NnNetConfigBuilder(NnUint nNodes, NnUint nBatches) {
        this->nNodes = nNodes;
        this->nBatches = nBatches;
    }

    NnUint addPipe(const char *name, NnSize2D size) {
        NnUint pipeIndex = pipes.size();
        pipes.push_back({ cloneString(name), size });
        return pipeIndex;
    }

    void addPreSync(NnUint pipeIndex) {
        preSyncs.push_back({ pipeIndex });
    }

    NnNetConfig build() {
        NnNetConfig config;
        config.nNodes = nNodes;
        config.nBatches = nBatches;
        config.nPipes = pipes.size();
        config.pipes = new NnPipeConfig[config.nPipes];
        std::copy(pipes.begin(), pipes.end(), config.pipes);
        config.nPreSyncs = preSyncs.size();
        if (config.nPreSyncs > 0) {
            config.preSyncs = new NnPreSyncConfig[config.nPreSyncs];
            std::copy(preSyncs.begin(), preSyncs.end(), config.preSyncs);
        } else {
            config.preSyncs = nullptr;
        }
        return config;
    }
};

class NnNodeConfigBuilder {
public:
    NnUint nodeIndex;
    std::list<NnBufferConfig> buffers;
    std::list<NnSegmentConfig> segments;

    NnNodeConfigBuilder(NnUint nodeIndex) {
        this->nodeIndex = nodeIndex;
    }

    NnUint addBuffer(const char *name, NnSize2D size) {
        NnUint bufferIndex = buffers.size();
        buffers.push_back({ cloneString(name), size });
        return bufferIndex;
    }

    void addSegment(NnSegmentConfig segment) {
        segments.push_back(segment);
    }

    NnNodeConfig build() {
        NnNodeConfig config;
        config.nodeIndex = nodeIndex;
        config.nBuffers = buffers.size();
        if (config.nBuffers > 0) {
            config.buffers = new NnBufferConfig[config.nBuffers];
            std::copy(buffers.begin(), buffers.end(), config.buffers);
        } else {
            config.buffers = nullptr;
        }

        config.nSegments = segments.size();
        assert(config.nSegments > 0);
        config.segments = new NnSegmentConfig[config.nSegments];
        std::copy(segments.begin(), segments.end(), config.segments);
        return config;
    }
};

class NnSegmentConfigBuilder {
private:
    std::list<NnOpConfig> ops;
    std::list<NnSyncConfig> syncs;

public:
    template <typename T>
    void addOp(NnOpCode code, const char *name, NnUint index, NnPointerConfig input, NnPointerConfig output, NnSize2D weightSize, T config) {
        NnUint configSize = sizeof(T);
        NnByte *configCopy = new NnByte[configSize];
        std::memcpy(configCopy, &config, configSize);
        ops.push_back({
            code,
            cloneString(name),
            index,
            input,
            output,
            weightSize,
            configCopy,
            configSize
        });
    };

    void addSync(NnUint pipeIndex, NnSyncType syncType) {
        syncs.push_back({ pipeIndex, syncType });
    }

    NnSegmentConfig build() {
        NnSegmentConfig segment;
        segment.nOps = ops.size();
        if (segment.nOps > 0) {
            segment.ops = new NnOpConfig[segment.nOps];
            std::copy(ops.begin(), ops.end(), segment.ops);
        }
        segment.nSyncs = syncs.size();
        if (segment.nSyncs > 0) {
            segment.syncs = new NnSyncConfig[segment.nSyncs];
            std::copy(syncs.begin(), syncs.end(), segment.syncs);
        }
        return segment;
    }
};

#endif