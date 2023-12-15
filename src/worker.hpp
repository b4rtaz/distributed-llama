#include "transformer.hpp"

#ifndef worker_hpp
#define worker_hpp

#define ACTION_HELLO 0
#define ACTION_CREATE_FRAGMENT 1
#define ACTION_FORWARD_FRAGMENT 2
#define ACTION_SEND_BUFFER 3

class WorkerRemoteClient: public RemoteClient {
private:
    int sliceCount;
    int* clientSockets;
public:
    WorkerRemoteClient(TransformerSpec* spec, char** hosts, int* ports);
    void createFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type, char* weights, int bytes);
    void forwardFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type);
    void sendBuffer(uint8_t sliceIndex, uint8_t bufferIndex, char* data, int bytes);
    void readBuffer(uint8_t sliceIndex, uint8_t bufferIndex, char* data, int bytes);
private:
    void sendBytes(uint8_t sliceIndex, void* data, int bytes);
};

struct WorkerLayer {
    NativeTransformerBlockQkv* qkv;
    NativeTransformerBlockAtt* att;
    NativeTransformerBlockFfn* ffn;
    NativeTransformerBlockFfn2* ffn2;
};

class Worker {
public:
    static void serve(int port);

private:
    int clientSocket;
    SharedBuffer* buffer;
    TransformerState* state;
    WorkerLayer* layers;
    TransformerSpec spec;
    uint8_t sliceIndex;

public:
    Worker(int clientSocket);
    void readSocket(void* data, int bytes);
    void writeSocket(void* data, int bytes);
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
    char* getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    char* getUnitBuffer(uint8_t bufferIndex);
    void waitForSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
    void sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex);
};

#endif
