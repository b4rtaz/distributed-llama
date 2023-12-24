#include "transformer.hpp"

#ifndef worker_hpp
#define worker_hpp

struct WorkerLayer {
    NativeTransformerBlockQkv* qkv;
    NativeTransformerBlockAtt* att;
    NativeTransformerBlockFfn* ffn;
    NativeTransformerBlockFfn2* ffn2;
};

class Worker {
public:
    static void serve(TransformerConfig* config, int port);

private:
    int clientSocket;
    SharedBuffer* buffer;
    TransformerState* state;
    WorkerLayer* layers;
    TransformerConfig* config;
    TransformerSpec spec;
    uint8_t sliceIndex;

public:
    Worker(TransformerConfig* config, int clientSocket);
    void readSocket(void* data, size_t bytes);
    void writeSocket(void* data, size_t bytes);
    void listen();
    void handleHello();
    void handleCreateFragment();
    void handleSendBuffer();
    void handleForwardFragment();
};

class WorkerTransformerState: public TransformerState {
private:
    SharedBuffer* buffer;
    Worker* worker;
public:
    WorkerTransformerState(SharedBuffer* buffer, Worker* worker);
    bool isRemote();
    char* getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    char* getUnitBuffer(uint8_t bufferIndex);
    void readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
};

#endif
