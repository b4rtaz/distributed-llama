#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <cassert>
#include <arpa/inet.h>
#include "funcs.hpp"
#include "worker.hpp"

#define SOCKET_CHUNK_SIZE 256

//
// WorkerRemoteClient
//

WorkerRemoteClient::WorkerRemoteClient(TransformerSpec* spec, char** hosts, int* ports) {
    this->sliceCount = spec->sliceCount;
    this->clientSockets = new int[spec->sliceCount];

    struct sockaddr_in clientAddr;
    for (uint8_t s = 0; s < sliceCount; s++) {
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
            printf("Cannot connect to %s:%d (%s)\n", host, port, strerror(errno));
            exit(EXIT_FAILURE);
        }

        printf("Connected to %s:%d\n", host, port);
        this->clientSockets[s] = clientSocket;

        uint8_t header[] = { ACTION_HELLO, s };
        sendBytes(s, header, sizeof(header));
        sendBytes(s, (void*)spec, sizeof(TransformerSpec));
    }
}

void WorkerRemoteClient::createFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type, char* weights, int bytes) {
    uint8_t header[] = { ACTION_CREATE_FRAGMENT, layerIndex, type };
    sendBytes(sliceIndex, header, sizeof(header));
    sendBytes(sliceIndex, (void*)&bytes, sizeof(int));
    sendBytes(sliceIndex, weights, bytes);
}

void WorkerRemoteClient::forwardFragment(uint8_t sliceIndex, uint8_t layerIndex, uint8_t type) {
    uint8_t header[] = { ACTION_FORWARD_FRAGMENT, layerIndex, type };
    sendBytes(sliceIndex, header, sizeof(header));
}

void WorkerRemoteClient::sendBuffer(uint8_t sliceIndex, uint8_t bufferIndex, char* data, int bytes) {
    uint8_t header[] = { ACTION_SEND_BUFFER, bufferIndex };
    sendBytes(sliceIndex, header, sizeof(header));
    sendBytes(sliceIndex, data, bytes);
}

void WorkerRemoteClient::readBuffer(uint8_t sliceIndex, uint8_t bufferIndex, char* data, int bytes) {
    int clientSocket = this->clientSockets[sliceIndex];
    uint bindex;
    int s1 = recv(clientSocket, (void*)&bindex, sizeof(uint8_t), 0);
    if (s1 != sizeof(uint8_t)) {
        printf("Error receiving buffer index\n");
        exit(EXIT_FAILURE);
    }
    if (bindex != bufferIndex) {
        printf("Error receiving buffer index %d, expected %d\n", bindex, bufferIndex);
        exit(EXIT_FAILURE);
    }
    int chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    int offset = 0;
    while (offset < bytes) {
        int chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int s2 = recv(clientSocket, (char*)data + offset, chunk, 0);
        if (s2 != chunk) {
            printf("Error receiving buffer data\n");
            exit(EXIT_FAILURE);
        }
        offset += chunk;
    }
}

void WorkerRemoteClient::sendBytes(uint8_t sliceIndex, void* data, int bytes) {
    int clientSocket = this->clientSockets[sliceIndex];
    int chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    int offset = 0;
    while (offset < bytes) {
        int chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int sendResult = send(clientSocket, (char*)data + offset, chunk, 0);
        if (sendResult != chunk) {
            printf("Error sending data\n");
            exit(EXIT_FAILURE);
        }
        offset += chunk;
    }
}

//
// Worker
//

Worker::Worker(int clientSocket) {
    this->clientSocket = clientSocket;
}

void Worker::readSocket(void* data, int bytes) {
    int chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    int offset = 0;
    while (offset < bytes) {
        int chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int recvStatus = recv(clientSocket, (char*)data + offset, chunk, 0);
        if (recvStatus != chunk) {
            printf("Error receiving data\n");
            exit(EXIT_FAILURE);
        }
        offset += chunk;
    }
}

void Worker::writeSocket(void* data, int bytes) {
    int chunkSize = bytes > SOCKET_CHUNK_SIZE ? SOCKET_CHUNK_SIZE : bytes;
    int offset = 0;
    while (offset < bytes) {
        int chunk = bytes - offset;
        if (chunk > chunkSize) {
            chunk = chunkSize;
        }
        int sendResult = send(clientSocket, (char*)data + offset, chunk, 0);
        if (sendResult != chunk) {
            printf("Error sending data\n");
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
    int bytes;
    readSocket((void*)&layerIndex, sizeof(uint8_t));
    readSocket((void*)&type, sizeof(uint8_t));
    readSocket((void*)&bytes, sizeof(int));

    switch (type) {
    case TRANSFORMER_BLOCK_QKV:
    {
        NativeTransformerBlockQkv* fragment = new NativeTransformerBlockQkv(layerIndex, sliceIndex, &spec, state);
        assert(fragment->qSlice->weights0Bytes + fragment->kSlice->weights0Bytes + fragment->vSlice->weights0Bytes == bytes);
        readSocket(fragment->qWeights0, fragment->qSlice->weights0Bytes);
        readSocket(fragment->kWeights0, fragment->kSlice->weights0Bytes);
        readSocket(fragment->vWeights0, fragment->vSlice->weights0Bytes);
        printf("Fragment qkv, layer=%d, weights=%d bytes\n", layerIndex, bytes);
        layers[layerIndex].qkv = fragment;
    }
    break;
    case TRANSFORMER_BLOCK_ATT:
    {
        NativeTransformerBlockAtt* fragment = new NativeTransformerBlockAtt(layerIndex, sliceIndex, &spec, state);
        assert(fragment->woSlice->weights0Bytes == bytes);
        readSocket(fragment->woWeights0, fragment->woSlice->weights0Bytes);
        printf("Fragment att, layer=%d, weights=%d bytes\n", layerIndex, bytes);
        layers[layerIndex].att = fragment;
    }
    break;
    case TRANSFORMER_BLOCK_FFN:
    {
        NativeTransformerBlockFfn* fragment = new NativeTransformerBlockFfn(layerIndex, sliceIndex, &spec, state);
        assert(fragment->w1Slice->weights0Bytes + fragment->w3Slice->weights0Bytes == bytes);
        readSocket(fragment->w1Weights0, fragment->w1Slice->weights0Bytes);
        readSocket(fragment->w3Weights0, fragment->w3Slice->weights0Bytes);
        printf("Fragment ffn, layer=%d, weights=%d bytes\n", layerIndex, bytes);
        layers[layerIndex].ffn = fragment;
    }
    break;
    case TRANSFORMER_BLOCK_FFN2:
    {
        NativeTransformerBlockFfn2* fragment = new NativeTransformerBlockFfn2(layerIndex, sliceIndex, &spec, state);
        assert(fragment->w2Slice->weights0Bytes == bytes);
        readSocket(fragment->w2Weights0, fragment->w2Slice->weights0Bytes);
        printf("Fragment ffn2, layer=%d, weights=%d bytes\n", layerIndex, bytes);
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

    int slices = buffer->getSlices(bufferIndex);
    int bytes = buffer->getBytes(bufferIndex);
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

    long t0 = timeMs();
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
    long t1 = timeMs();
    printf("Processed fragment %2d/%d in %3ldms\n", layerIndex, type, t1 - t0);
}

void Worker::serve(int port) {
    const char* host = "127.0.0.1";
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

    int listenResult = ::listen(serverSocket, 10);
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

        Worker worker(clientSocket);
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

void WorkerTransformerState::waitForSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    printf("Unexpected call to waitForSlicedBuffer\n");
    exit(EXIT_FAILURE);
}

void WorkerTransformerState::sendSlicedBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    char* data = buffer->getSliced(bufferIndex, sliceIndex);
    int bytes = buffer->getBytes(bufferIndex) / buffer->getSlices(bufferIndex);
    worker->writeSocket((void*)&bufferIndex, sizeof(uint8_t));
    worker->writeSocket(data, bytes);
}

void WorkerTransformerState::sendUnitBuffer(uint8_t bufferIndex, uint8_t sliceIndex) {
    printf("Unexpected call to sendUnitBuffer\n");
    exit(EXIT_FAILURE);
}