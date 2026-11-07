#include "graph.hpp"
#include "../mmap.hpp"
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>

const LlmModel &resolveLlmModel(int magic) {
    if (magic == 0xABCD00 || magic == 0xABCD01)
        throw std::runtime_error("Old model format is not supported");
    const LlmModel *models[] = {
        &getLlamaLlmModel(),
        &getQwen35LlmModel(),
    };
    for (const LlmModel *model : models) {
        if (model->isMagicSupported(magic))
            return *model;
    }
    throw std::runtime_error("Unsupported model magic: " + std::to_string(magic));
}

int loadLlmMagic(const char *path) {
    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *file = fdPtr.get();
    if (file == NULL)
        throw std::runtime_error(std::string("Cannot open model file (") + path + std::string("): ") + std::strerror(errno));
    int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1)
        throw std::runtime_error("Cannot read magic value");
    return magic;
}

std::unique_ptr<LlmHeader> loadLlmHeader(
    const LlmModel &model, int magic, const char *path, const NnUint maxSeqLen, NnFloatType syncType) {
    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *file = fdPtr.get();
    if (file == NULL)
        throw std::runtime_error(std::string("Cannot open model file (") + path + std::string("): ") + std::strerror(errno));

    fseek(file, sizeof(int), SEEK_SET);

    std::unique_ptr<LlmHeader> header = model.loadHeader(file, magic, maxSeqLen, syncType);
    header->magic = magic;
    header->syncType = syncType;
    header->fileSize = (NnSize)seekToEnd(file);
    return header;
}

LlmGraph::~LlmGraph() {
    if (nodeConfigs != nullptr) {
        for (NnUint nodeIndex = 0u; nodeIndex < netConfig.nNodes; nodeIndex++)
            releaseNodeConfig(&nodeConfigs[nodeIndex]);
        delete[] nodeConfigs;
    }
    releaseNetConfig(&netConfig);
}

