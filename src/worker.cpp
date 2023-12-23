#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <unistd.h>
#include <cstring>
#include <errno.h>
#include <cassert>
#include <arpa/inet.h>
#include "funcs.hpp"
#include "worker.hpp"

#define SOCKET_LAST_ERROR strerror(errno)
#define SOCKET_CHUNK_SIZE 128

//
// WorkerRemoteClient
//

WorkerRemoteClient::WorkerRemoteClient(TransformerSpec* spec, char** hosts, int* ports) {
    sliceCount = spec->sliceCount;
    clientSockets = new int[spec->sliceCount];
    waitBufferTime = new long[spec->sliceCount];
    sendBufferTime = new long[spec->sliceCount];
    readBufferTime = new long[spec->sliceCount];

    struct sockaddr_in clientAddr;
    for (uint8_t s = 0; s < sliceCount; s++) {
        waitBufferTime[s] = 0;
        sendBufferTime[s] = 0;
        readBufferTime[s] = 0;

        char* host = hosts[s];
        int port = ports[s];

        memset(&clientAddr, 0, sizeof(clientAddr));
        clientAddr.sin_family = AF_INET;
        clientAddr.sin_addr.s_addr = inet_addr(host);
        clientAddr.sin_port = htons(port);

        int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (clientSocket < 0) {
            printf("Error creating socket\n");
            exit(EXIT_FAILURE);
        }

        int connectResult = connect(clientSocket, (struct sockaddr*)&clientAddr, sizeof(clientAddr));
        if (connectResult != 0) {
            printf("Cannot connect to %s:%d (%s)\n", host, port, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }

        printf("Connected to %s:%d\n", host, port);
        this->clientSockets[s] = clientSocket;

        uint8_t header[] = { ACTION_HELLO, s };
        sendBytes(s, header, sizeof(header));
        sendBytes(s, (void*)spec, sizeof(TransformerSpec));
    }
}

WorkerRemoteClient::~WorkerRemoteClient() {
    delete[] clientSockets;
    delete[] waitBufferTime;
    delete[] sendBufferTime;
    delete[] readBufferTime;
}

void WorkerRemoteClient::createFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type, char* weights, size_t bytes) {
    uint8_t header[] = { ACTION_CREATE_FRAGMENT, layerIndex, type };
    sendBytes(sliceIndex, header, sizeof(header));
    sendBytes(sliceIndex, &bytes, sizeof(size_t));
    sendBytes(sliceIndex, weights, bytes);
}

void WorkerRemoteClient::forwardFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type) {
    uint8_t header[] = { ACTION_FORWARD_FRAGMENT, layerIndex, type };
    sendBytes(sliceIndex, header, sizeof(header));
}

void WorkerRemoteClient::sendBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes) {
    uint8_t header[] = { ACTION_SEND_BUFFER, bufferIndex };
    long t0 = timeMs();
    sendBytes(sliceIndex, header, sizeof(header));
    sendBytes(sliceIndex, data, bytes);
    long t1 = timeMs();
    sendBufferTime[sliceIndex] += t1 - t0;
}

void WorkerRemoteClient::readBuffer(uint8_t sliceIndex, uint8_t bufferIndex, void* data, size_t bytes) {
    int clientSocket = this->clientSockets[sliceIndex];

    long t0 = timeMs();
    uint8_t header[2];
    readBytes(sliceIndex, (void*)&header, sizeof(header));
    long t1 = timeMs();
    readBytes(sliceIndex, data, bytes);
    long t2 = timeMs();

    waitBufferTime[sliceIndex] += t1 - t0;
    readBufferTime[sliceIndex] += t2 - t1;
}

void WorkerRemoteClient::sendBytes(uint8_t sliceIndex, void* data, size_t bytes) {
    int clientSocket = this->clientSockets[sliceIndex];
    size_t chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    size_t offset = 0;
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int s = send(clientSocket, (char*)data + offset, chunk, 0);
        if (s <= 0) {
            printf("Error sending data %d (%s)\n", s, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        offset += s;
    }
}

void WorkerRemoteClient::readBytes(uint8_t sliceIndex, void* data, size_t bytes) {
    int clientSocket = this->clientSockets[sliceIndex];
    size_t chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    size_t offset = 0;
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int r = recv(clientSocket, (char*)data + offset, chunk, 0);
        if (r <= 0) {
            printf("Error receiving buffer data %d (%s)\n", r, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        offset += r;
    }
}

void WorkerRemoteClient::dumpStatistics() {
    printf("⌛ ");
    for (size_t s = 0; s < sliceCount; s++) {
        printf("%zu: %3ldms/%3ldms/%4ldms ", s, sendBufferTime[s], readBufferTime[s], waitBufferTime[s]);
        sendBufferTime[s] = 0;
        readBufferTime[s] = 0;
        waitBufferTime[s] = 0;
    }
    printf("\n");
}

//
// Worker
//

Worker::Worker(TransformerConfig* config, int clientSocket) {
    this->config = config;
    this->clientSocket = clientSocket;
}

void Worker::readSocket(void* data, size_t bytes) {
    size_t chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    size_t offset = 0;
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int r = recv(clientSocket, (char*)data + offset, chunk, 0);
        if (r <= 0) {
            printf("Error receiving data %d (%s)\n", r, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        offset += r;
    }
}

void Worker::writeSocket(void* data, size_t bytes) {
    size_t chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    size_t offset = 0;
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int s = send(clientSocket, (char*)data + offset, chunk, 0);
        if (s <= 0) {
            printf("Error sending data %d (%s)\n", s, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        offset += chunk;
    }
}

void Worker::listen() {
    uint8_t action;
    for (;;) {
        readSocket((void*)&action, sizeof(uint8_t));
        switch (action)
        {
        case ACTION_HELLO:
            handleHello();
            break;
        case ACTION_CREATE_FRAGMENT:
            handleCreateFragment();
            break;
        case ACTION_SEND_BUFFER:
            handleSendBuffer();
            break;
        case ACTION_FORWARD_FRAGMENT:
            handleForwardFragment();
            break;
        default:
            printf("Unknown action: %d\n", action);
            exit(EXIT_FAILURE);
        }
    }
}

void Worker::handleHello() {
    readSocket((void*)&sliceIndex, sizeof(uint8_t));
    readSocket((void*)&spec, sizeof(TransformerSpec));
    layers = new WorkerLayer[spec.nLayers];
    buffer = initSharedBuffer(&spec);
    state = new WorkerTransformerState(buffer, this);
    printf("Set slice index %d\n", sliceIndex);
}

void Worker::handleCreateFragment() {
    uint8_t layerIndex;
    uint8_t type;
    size_t bytes;
    readSocket((void*)&layerIndex, sizeof(uint8_t));
    readSocket((void*)&type, sizeof(uint8_t));
    readSocket((void*)&bytes, sizeof(size_t));

    switch (type) {
    case TRANSFORMER_BLOCK_QKV:
    {
        NativeTransformerBlockQkv* fragment = new NativeTransformerBlockQkv(layerIndex, sliceIndex, &spec, config, state);
        assert(fragment->qSlice->weights0Bytes + fragment->kSlice->weights0Bytes + fragment->vSlice->weights0Bytes == bytes);
        readSocket(fragment->qWeights0, fragment->qSlice->weights0Bytes);
        readSocket(fragment->kWeights0, fragment->kSlice->weights0Bytes);
        readSocket(fragment->vWeights0, fragment->vSlice->weights0Bytes);
        printf("Fragment qkv, layer=%d, weights=%zu bytes\n", layerIndex, bytes);
        layers[layerIndex].qkv = fragment;
    }
    break;
    case TRANSFORMER_BLOCK_ATT:
    {
        NativeTransformerBlockAtt* fragment = new NativeTransformerBlockAtt(layerIndex, sliceIndex, &spec, config, state);
        assert(fragment->woSlice->weights0Bytes == bytes);
        readSocket(fragment->woWeights0, fragment->woSlice->weights0Bytes);
        printf("Fragment att, layer=%d, weights=%zu bytes\n", layerIndex, bytes);
        layers[layerIndex].att = fragment;
    }
    break;
    case TRANSFORMER_BLOCK_FFN:
    {
        NativeTransformerBlockFfn* fragment = new NativeTransformerBlockFfn(layerIndex, sliceIndex, &spec, config, state);
        assert(fragment->w1Slice->weights0Bytes + fragment->w3Slice->weights0Bytes == bytes);
        readSocket(fragment->w1Weights0, fragment->w1Slice->weights0Bytes);
        readSocket(fragment->w3Weights0, fragment->w3Slice->weights0Bytes);
        printf("Fragment ffn, layer=%d, weights=%zu bytes\n", layerIndex, bytes);
        layers[layerIndex].ffn = fragment;
    }
    break;
    case TRANSFORMER_BLOCK_FFN2:
    {
        NativeTransformerBlockFfn2* fragment = new NativeTransformerBlockFfn2(layerIndex, sliceIndex, &spec, config, state);
        assert(fragment->w2Slice->weights0Bytes == bytes);
        readSocket(fragment->w2Weights0, fragment->w2Slice->weights0Bytes);
        printf("Fragment ffn2, layer=%d, weights=%zu bytes\n", layerIndex, bytes);
        layers[layerIndex].ffn2 = fragment;
    }
    break;
    default:
        printf("Unknown fragment type %d\n", type);
        exit(EXIT_FAILURE);
    }
}

void Worker::handleSendBuffer() {
    uint8_t bufferIndex;
    readSocket((void*)&bufferIndex, sizeof(uint8_t));

    size_t slices = buffer->getSlices(bufferIndex);
    size_t bytes = buffer->getBytes(bufferIndex);
    char* data;
    if (slices == SLICES_UNIT) {
        data = buffer->getUnit(bufferIndex);
    } else {
        bytes = bytes / slices;
        data = buffer->getSliced(bufferIndex, sliceIndex);
    }
    readSocket(data, bytes);
}

void Worker::handleForwardFragment() {
    uint8_t layerIndex;
    uint8_t type;
    readSocket((void*)&layerIndex, sizeof(uint8_t));
    readSocket((void*)&type, sizeof(uint8_t));

    switch (type) {
    case TRANSFORMER_BLOCK_QKV:
        layers[layerIndex].qkv->beginForwarding();
        break;
    case TRANSFORMER_BLOCK_ATT:
        layers[layerIndex].att->beginForwarding();
        break;
    case TRANSFORMER_BLOCK_FFN:
        layers[layerIndex].ffn->beginForwarding();
        break;
    case TRANSFORMER_BLOCK_FFN2:
        layers[layerIndex].ffn2->beginForwarding();
        break;
    default:
        printf("Unknown fragment type %d\n", type);
        exit(EXIT_FAILURE);
    }
}

void Worker::serve(TransformerConfig* config, int port) {
    const char* host = "0.0.0.0";
    int serverSocket;
    struct sockaddr_in serverAddr;
    struct sockaddr_in clientAddr;

    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        printf("Error creating socket\n");
        exit(EXIT_FAILURE);
    }

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        printf("Cannot bind %s:%d\n", host, port);
        exit(EXIT_FAILURE);
    }

    int listenResult = ::listen(serverSocket, 1);
    if (listenResult != 0) {
        printf("Cannot listen %s:%d\n", host, port);
        exit(EXIT_FAILURE);
    }
    printf("Listening on %s:%d...\n", host, port);

    socklen_t clientAddrSize = sizeof(clientAddr);
    for(;;) {
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
        if (clientSocket < 0) {
            printf("Error accepting connection\n");
            exit(EXIT_FAILURE);
        }

        Worker worker(config, clientSocket);
        printf("Client connected\n");
        worker.listen();
        printf("Client disconnected\n");
        
        close(clientSocket);
    }
}

//
// WorkerTransformerState
//

WorkerTransformerState::WorkerTransformerState(SharedBuffer* buffer, Worker* worker) {
    this->buffer = buffer;
    this->worker = worker;
}

char* WorkerTransformerState::getSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    return buffer->getSliced(bufferIndex, sliceIndex);
}

char* WorkerTransformerState::getUnitBuffer(uint8_t bufferIndex) {
    return buffer->getUnit(bufferIndex);
}

void WorkerTransformerState::readSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    printf("Unexpected call to readSlicedBuffer\n");
    exit(EXIT_FAILURE);
}

void WorkerTransformerState::sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    char* data = buffer->getSliced(bufferIndex, sliceIndex);
    size_t bytes = buffer->getBytes(bufferIndex) / buffer->getSlices(bufferIndex);
    uint8_t header[] = { ACTION_SEND_BUFFER, bufferIndex };
    worker->writeSocket((void*)&header, sizeof(header));
    worker->writeSocket(data, bytes);
}

void WorkerTransformerState::sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    printf("Unexpected call to sendUnitBuffer\n");
    exit(EXIT_FAILURE);
}