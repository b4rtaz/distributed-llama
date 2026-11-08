#ifndef LLM_GRAPH_HPP
#define LLM_GRAPH_HPP

#include "../nn/nn-core.hpp"
#include "../nn/nn-executor.hpp"
#include "../nn/nn-network.hpp"
#include <cstdio>
#include <climits>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

enum LlmMagic {
    LLM_MAGIC_LLAMA = 0xA00ABCD,
    LLM_MAGIC_QWEN35 = 0xA00ABCE,
};

inline float convertNormEpsilon(int value) {
    if (value == 5) return 1e-5f;
    if (value == 6) return 1e-6f;
    throw std::runtime_error("Unsupported norm epsilon");
}

struct LlmHeader {
    virtual ~LlmHeader() {}

    int magic;
    NnSize headerSize;
    NnSize fileSize;
    int version;
    NnUint nKvHeads;
    NnUint origSeqLen; // Original model context length
    NnUint seqLen; // Limited context length by the `--max-seq-len` argument
    NnUint vocabSize;
    NnFloatType weightType;
    NnFloatType syncType;
};

class LlmGraph {
public:
    LlmHeader *header;
    NnNetConfig netConfig;
    NnNodeConfig *nodeConfigs;
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint outputPipeIndex;

    LlmGraph()
        : header(nullptr), netConfig(), nodeConfigs(nullptr),
          positionPipeIndex(UINT_MAX), tokenPipeIndex(UINT_MAX), outputPipeIndex(UINT_MAX) {}
    virtual ~LlmGraph();

private:
    LlmGraph(const LlmGraph &);
};

class LlmModel {
public:
    const std::vector<int> magicNumbers;

    explicit LlmModel(std::initializer_list<int> supportedMagicNumbers)
        : magicNumbers(supportedMagicNumbers) {}
    virtual ~LlmModel() {}

    bool isMagicSupported(int magic) const {
        for (int supportedMagic : magicNumbers) {
            if (supportedMagic == magic)
                return true;
        }
        return false;
    }

    virtual std::unique_ptr<LlmHeader> loadHeader(
        FILE *file, int magic, NnUint maxSeqLen, NnFloatType syncType) const = 0;
    virtual void printHeader(const LlmHeader *header) const = 0;
    virtual std::unique_ptr<LlmGraph> buildGraph(
        LlmHeader *header, NnUint nNodes, NnUint nBatches) const = 0;
    virtual void loadGraph(const char *path, LlmGraph *graph, NnRootWeightLoader *loader) const = 0;
};

const LlmModel &getLlamaLlmModel();
const LlmModel &getQwen35LlmModel();

int loadLlmMagic(const char *path);
const LlmModel &resolveLlmModel(int magic);
std::unique_ptr<LlmHeader> loadLlmHeader(
    const LlmModel &model, int magic, const char *path, NnUint maxSeqLen, NnFloatType syncType);

#endif
