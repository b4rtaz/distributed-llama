#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h> // For inet_addr and other functions
#include <windows.h>  // For SSIZE_T
typedef SSIZE_T ssize_t;
#define close closesocket
#else
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif
#include "nn-network.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

#define ACK 23571113
#define ONE_MB 1048576

static inline bool isEagainError() {
    #ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
    #else
    return SOCKET_LAST_ERRCODE == EAGAIN;
    #endif
}

static inline void setNonBlocking(int socket, bool enabled) {
#ifdef _WIN32
    u_long mode = enabled ? 1 : 0;
    if (ioctlsocket(socket, FIONBIO, &mode) != 0) {
        throw std::runtime_error("Error setting socket to non-blocking");
    }
#else
    int flags = fcntl(socket, F_GETFL, 0);
    if (enabled) {
        flags |= O_NONBLOCK;
    } else {
        flags = flags & (~O_NONBLOCK);
    }
    if (fcntl(socket, F_SETFL, flags) < 0)
        throw std::runtime_error("Error setting socket to non-blocking");
#endif
}

static inline void setNoDelay(int socket) {
    int flag = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error setting socket to no-delay");
}

static inline void setQuickAck(int socket) {
#ifndef _WIN32
#ifdef TCP_QUICKACK
    int value = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_QUICKACK, (char*)&value, sizeof(int)) < 0)
        throw std::runtime_error("Error setting quick ack");
#endif
#endif
}

void setReuseAddr(int socket) {
    int opt = 1;
    #ifdef _WIN32
    int iresult = setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
    if (iresult == SOCKET_ERROR) {
        closesocket(socket);
        throw std::runtime_error("setsockopt failed: " + std::to_string(WSAGetLastError()));
    }
    #else
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(socket);
        throw std::runtime_error("setsockopt failed: " + std::string(strerror(errno)));
    }
    #endif
}

void writeSocket(int socket, const void *data, size_t size) {
    while (size > 0) {
        ssize_t s = send(socket, (const char*)data, size, 0);
        if (s < 0) {
            if (isEagainError()) {
                continue;
            }
            throw NnWriteNetworkException(0, "Error writing to socket");
        } else if (s == 0) {
            throw NnWriteNetworkException(0, "Socket closed");
        }
        size -= s;
        data = (const char*)data + s;
    }
}

static inline bool tryReadSocket(int socket, void *data, size_t size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    size_t s = size;
    while (s > 0) {
        ssize_t r = recv(socket, (char*)data, s, 0);
        if (r < 0) {
            if (isEagainError()) {
                if (s == size && maxAttempts > 0) {
                    maxAttempts--;
                    if (maxAttempts == 0) {
                        return false;
                    }
                }
                continue;
            }
            throw NnReadNetworkException(0, "Error reading from socket");
        } else if (r == 0) {
            throw NnReadNetworkException(0, "Socket closed");
        }
        data = (char*)data + r;
        s -= r;
    }
    return true;
}

void readSocket(int socket, void *data, size_t size) {
    if (!tryReadSocket(socket, data, size, 0)) {
        throw std::runtime_error("Error reading from socket");
    }
}

static void readAckPacket(int socket) {
    NnSize packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != ACK)
        throw std::runtime_error("Invalid ack packet");
}

static void writeAckPacket(int socket) {
    NnSize packet = ACK;
    writeSocket(socket, &packet, sizeof(packet));
}

static inline int connectSocket(char *host, int port) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(host);
    addr.sin_port = htons(port);

    int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
        throw std::runtime_error("Cannot create socket");

    int connectResult = ::connect(sock, (struct sockaddr*)&addr, sizeof(addr));
    if (connectResult != 0) {
        printf("Cannot connect to %s:%d (%s)\n", host, port, SOCKET_LAST_ERROR);
        throw std::runtime_error("Cannot connect");
    }

    setNoDelay(sock);
    setQuickAck(sock);
    return sock;
}

int createServerSocket(int port) {
    const char *host = "0.0.0.0";
    struct sockaddr_in serverAddr;

    int serverSocket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket < 0)
        throw std::runtime_error("Cannot create socket");
    setReuseAddr(serverSocket);

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult;
    #ifdef _WIN32
    bindResult = bind(serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr));
    if (bindResult == SOCKET_ERROR) {
        int error = WSAGetLastError();
        closesocket(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::to_string(error));
    }
    #else
    bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        close(serverSocket);
        throw std::runtime_error("Cannot bind port: " + std::string(strerror(errno)));
    }
    #endif

    int listenResult = listen(serverSocket, SOMAXCONN);
    if (listenResult != 0) {
        #ifdef _WIN32
        closesocket(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::to_string(WSAGetLastError()));
        #else
        close(serverSocket);
        throw std::runtime_error("Cannot listen on port: " + std::string(strerror(errno)));
        #endif
    }

    printf("Listening on %s:%d...\n", host, port);

    setNoDelay(serverSocket);
    setQuickAck(serverSocket);
    return serverSocket;
}

void closeServerSocket(int serverSocket) {
    shutdown(serverSocket, 2);
    close(serverSocket);
}

int acceptSocket(int serverSocket) {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0)
        throw std::runtime_error("Error accepting connection");
    setNoDelay(clientSocket);
    setQuickAck(clientSocket);
    return clientSocket;
}

void initSockets() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        throw std::runtime_error("WSAStartup failed: " + std::to_string(WSAGetLastError()));
    }
#endif
}

void cleanupSockets() {
#ifdef _WIN32
    WSACleanup();
#endif
}

NnReadNetworkException::NnReadNetworkException(int code, const char *message) {
    this->code = code;
    this->message = message;
}

NnWriteNetworkException::NnWriteNetworkException(int code, const char *message) {
    this->code = code;
    this->message = message;
}

std::unique_ptr<NnNetwork> NnNetwork::serve(int port) {
    int serverSocket = createServerSocket(port);

    NnSize nSockets;
    NnSize nodeIndex;
    int rootSocket = acceptSocket(serverSocket);
    printf("⭕ The root node has connected\n");

    readSocket(rootSocket, &nSockets, sizeof(nSockets));
    NnSize nNodes = nSockets - 1; // nSockets - 1 root node
    printf("⭕ nNodes: %d\n", nNodes);
    readSocket(rootSocket, &nodeIndex, sizeof(nodeIndex));
    printf("⭕ NodeIndex: %d\n", nodeIndex);

    int *sockets = new int[nSockets];
    sockets[0] = rootSocket;
    char* *hosts = new char*[nNodes];
    int *ports = new int[nNodes];
    printf("⭕ Socket[0]: accepted root node\n");

    size_t hostLen;
    for (NnSize i = 0; i < nNodes; i++) {
        readSocket(rootSocket, &hostLen, sizeof(hostLen));
        hosts[i] = new char[hostLen];
        readSocket(rootSocket, hosts[i], hostLen);
        readSocket(rootSocket, &ports[i], sizeof(ports[i]));
    }

    writeAckPacket(rootSocket);

    // We need to wait here until the root node will send a "root is ready" packet
    readAckPacket(rootSocket);

    for (NnSize i = 0; i < nNodes; i++) {
        NnSize socketIndex = i + 1;
        if (i >= nodeIndex) {
            printf("⭕ Socket[%d]: connecting to %s:%d worker\n", socketIndex, hosts[i], ports[i]);
            sockets[socketIndex] = connectSocket(hosts[i], ports[i]);
            printf("⭕ Socket[%d]: connected\n", socketIndex);
        } else {
            printf("⭕ Socket[%d]: wait for %s:%d worker\n", socketIndex, hosts[i], ports[i]);
            sockets[socketIndex] = acceptSocket(serverSocket);
            printf("⭕ Socket[%d]: accepted\n", socketIndex);
        }
    }

    for (NnSize i = 0; i < nNodes; i++)
        delete[] hosts[i];
    delete[] hosts;
    delete[] ports;

    shutdown(serverSocket, 2);
    close(serverSocket);
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(nSockets, sockets));
}

std::unique_ptr<NnNetwork> NnNetwork::connect(NnSize nSockets, char **hosts, NnSize *ports) {
    assert(nSockets > 0);

    int *sockets = new int[nSockets];
    struct sockaddr_in addr;
    NnSize confirmPacket;
    for (NnSize i = 0; i < nSockets; i++) {
        printf("⭕ Socket[%d]: connecting to %s:%d worker\n", i, hosts[i], ports[i]);
        int socket = connectSocket(hosts[i], ports[i]);
        sockets[i] = socket;
        writeSocket(socket, &nSockets, sizeof(nSockets));
        writeSocket(socket, &i, sizeof(i)); // send node index
        for (NnSize j = 0; j < nSockets; j++) {
            if (j == i)
                continue;
            size_t hostLen = strlen(hosts[j]) + 1;
            writeSocket(socket, &hostLen, sizeof(hostLen));
            writeSocket(socket, hosts[j], hostLen);
            writeSocket(socket, &ports[j], sizeof(ports[j]));
        }
        readAckPacket(socket);
        printf("⭕ Socket[%d]: connected\n", i);
    }
    for (NnSize i = 0; i < nSockets; i++) {
        writeAckPacket(sockets[i]);
    }
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(nSockets, sockets));
}

NnNetwork::NnNetwork(NnSize nSockets, int *sockets) {
    this->nSockets = nSockets;
    this->sockets = sockets;
    this->sentBytes.exchange(0);
    this->recvBytes.exchange(0);
}

NnNetwork::~NnNetwork() {
    for (NnSize i = 0; i < nSockets; i++) {
        shutdown(sockets[i], 2);
        close(sockets[i]);
    }
    delete[] sockets;
    printf("⭕ Network is closed\n");
}

void NnNetwork::setTurbo(bool enabled) {
    for (NnSize i = 0; i < nSockets; i++) {
        ::setNonBlocking(sockets[i], enabled);
    }
}

void NnNetwork::write(NnSize socketIndex, const void *data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    sentBytes += size;

    char *current = (char*)data;
    int s = sockets[socketIndex];
    for (size_t chunk = 0; chunk < size; chunk += ONE_MB) {
        size_t chunkSize = chunk + ONE_MB < size ? ONE_MB : size - chunk;
        writeSocket(s, current, chunkSize);
        current += chunkSize;
    }
}

void NnNetwork::read(NnSize socketIndex, void *data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    recvBytes += size;

    char *current = (char*)data;
    int s = sockets[socketIndex];
    for (size_t chunk = 0; chunk < size; chunk += ONE_MB) {
        size_t chunkSize = chunk + ONE_MB < size ? ONE_MB : size - chunk;
        readSocket(s, current, chunkSize);
        current += chunkSize;
    }
}

void NnNetwork::writeAck(NnSize socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    writeAckPacket(sockets[socketIndex]);
}

void NnNetwork::readAck(NnSize socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    readAckPacket(sockets[socketIndex]);
}

bool NnNetwork::tryReadWithMaxAttempts(NnSize socketIndex, void *data, size_t size, unsigned long maxAttempts) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    if (tryReadSocket(sockets[socketIndex], data, size, maxAttempts)) {
        recvBytes += size;
        return true;
    }
    return false;
}

void NnNetwork::writeMany(NnSize n, NnSocketIo *ios) {
    bool isWriting;
    size_t nBytes = 0;
    for (NnSize i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex >= 0 && io->socketIndex < nSockets);
        nBytes += io->size;
    }
    do {
        isWriting = false;
        for (NnSize i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isWriting = true;
                int socket = sockets[io->socketIndex];
                ssize_t s = send(socket, (const char*)io->data, io->size, 0);
                if (s < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnWriteNetworkException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (s == 0) {
                    throw NnWriteNetworkException(0, "Socket closed");
                }
                io->size -= s;
                io->data = (char*)io->data + s;
            }
        }
    } while (isWriting);
    sentBytes += nBytes;
}

void NnNetwork::writeAll(void *data, size_t size) {
    std::vector<NnSocketIo> ios(nSockets);
    for (NnSize i = 0; i < nSockets; i++) {
        NnSocketIo *io = &ios[i];
        io->socketIndex = i;
        io->data = data;
        io->size = size;
    }
    writeMany(nSockets, &ios[0]);
}

void NnNetwork::readMany(NnSize n, NnSocketIo *ios) {
    bool isReading;
    size_t nBytes = 0;
    for (NnSize i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex >= 0 && io->socketIndex < nSockets);
        nBytes += io->size;
    }
    do {
        isReading = false;
        for (NnSize i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isReading = true;
                int socket = sockets[io->socketIndex];
                ssize_t r = recv(socket, (char*)io->data, io->size, 0);
                if (r < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw NnReadNetworkException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (r == 0) {
                    throw NnReadNetworkException(0, "Socket closed");
                }
                io->size -= r;
                io->data = (char*)io->data + r;
            }
        }
    } while (isReading);
    recvBytes += nBytes;
}

void NnNetwork::getStats(size_t *sentBytes, size_t *recvBytes) {
    *sentBytes = this->sentBytes;
    *recvBytes = this->recvBytes;
    this->resetStats();
}

void NnNetwork::resetStats() {
    this->sentBytes.exchange(0);
    this->recvBytes.exchange(0);
}

static void syncWithRoot(NnNetwork *network, NnByte nodeIndex, NnByte *buffer, NnSize nBytes, NnSize nThreads, NnSize threadIndex) {
    if (nodeIndex == 0) {
        // root

        unsigned int nSocketsPerThread = network->nSockets / nThreads + (network->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSocketsPerThread == 0) return;

        std::vector<NnSocketIo> ios(nSocketsPerThread);
        for (int i = 0; i < nSocketsPerThread; i++) {
            ios[i].socketIndex = threadIndex + i * nThreads;
            ios[i].data = buffer;
            ios[i].size = nBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    } else {
        // worker

        if (threadIndex != 0) return;

        NnSocketIo ios;
        ios.data = buffer;
        ios.size = nBytes;
        ios.socketIndex = 0; // root
        network->readMany(1, &ios);
    }
}

static void syncNodeSlices(bool onlyFromWorkerToRoot, NnNetwork *network, NnSize nodeIndex, NnSize nNodes, NnByte *buffer, NnSize nBytes, NnSize nThreads, NnSize threadIndex) {
    bool isWorker = nodeIndex != 0;
    NnSize nSockets = onlyFromWorkerToRoot && isWorker ? 1 : network->nSockets;
    NnSize nSocketsPerThread = nSockets / nThreads + (nSockets % nThreads > threadIndex ? 1 : 0);
    if (nSocketsPerThread == 0) return;
    NnSize sliceBytes = nBytes / nNodes;

    std::unique_ptr<NnSocketIo> iosPtr(new NnSocketIo[nSocketsPerThread]);
    NnSocketIo *ios = iosPtr.get();

    if (!onlyFromWorkerToRoot || isWorker) {
        NnByte *mySliceData = &buffer[sliceBytes * nodeIndex];

        for (unsigned int i = 0; i < nSocketsPerThread; i++) {
            unsigned int socketIndex = threadIndex + i * nThreads;
            ios[i].socketIndex = socketIndex;
            ios[i].data = mySliceData;
            ios[i].size = sliceBytes;
        }
        network->writeMany(nSocketsPerThread, ios);
    }

    if (!onlyFromWorkerToRoot || !isWorker) {
        for (unsigned int i = 0; i < nSocketsPerThread; i++) {
            unsigned int socketIndex = threadIndex + i * nThreads;
            int sliceIndex = socketIndex >= nodeIndex ? socketIndex + 1 : socketIndex;
            NnByte *sliceData = &buffer[sliceBytes * sliceIndex];
            ios[i].socketIndex = socketIndex;
            ios[i].data = sliceData;
            ios[i].size = sliceBytes;
        }
        network->readMany(nSocketsPerThread, ios);
    }
}

NnNetworkNodeSynchronizer::NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    this->network = network;
    this->execution = execution;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
}

void NnNetworkNodeSynchronizer::sync(NnSize segmentIndex, NnSize nThreads, NnSize threadIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];

    for (NnSize syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
        NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
        NnByte *pipe = execution->pipes[syncConfig->pipeIndex];
        NnPipeConfig *pipeConfig = &netConfig->pipes[syncConfig->pipeIndex];
        NnSize batchBytes = getBytes(pipeConfig->size.floatType, pipeConfig->size.x);

        for (NnSize batchIndex = 0; batchIndex < execution->batchSize; batchIndex++) {
            NnByte *pipeBatch = &pipe[batchIndex * batchBytes];

            if (syncConfig->syncType == SYNC_WITH_ROOT) {
                syncWithRoot(network, nodeConfig->nodeIndex, pipeBatch, batchBytes, nThreads, threadIndex);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES) {
                syncNodeSlices(false, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, nThreads, threadIndex);
            } else if (syncConfig->syncType == SYNC_NODE_SLICES_EXCEPT_ROOT) {
                syncNodeSlices(true, network, nodeConfig->nodeIndex, netConfig->nNodes, pipeBatch, batchBytes, nThreads, threadIndex);
            } else {
                throw std::invalid_argument("Unknown sync type");
            }
        }
    }
}

static void writeString(NnNetwork *network, NnSize socketIndex, char *str) {
    NnSize bytes = std::strlen(str) + 1;
    network->write(socketIndex, &bytes, sizeof(NnSize));
    network->write(socketIndex, str, bytes);
}

static char *readString(NnNetwork *network, NnSize socketIndex) {
    NnSize bytes;
    network->read(socketIndex, &bytes, sizeof(NnSize));
    char *str = new char[bytes];
    network->read(socketIndex, str, bytes);
    return str;
}

NnRootConfigWriter::NnRootConfigWriter(NnNetwork *network) {
    this->network = network;
}

void NnRootConfigWriter::writeNet(NnSize socketIndex, NnNetConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nBatches, sizeof(config->nBatches));
    network->write(socketIndex, &config->nNodes, sizeof(config->nNodes));
    network->write(socketIndex, &config->nPipes, sizeof(config->nPipes));
    for (NnSize pipeIndex = 0; pipeIndex < config->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config->pipes[pipeIndex];
        network->write(socketIndex, &pipeConfig->size, sizeof(pipeConfig->size));
        writeString(network, socketIndex, pipeConfig->name);
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeNode(NnSize socketIndex, NnNodeConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nodeIndex, sizeof(config->nodeIndex));
    network->write(socketIndex, &config->nBuffers, sizeof(config->nBuffers));
    network->write(socketIndex, &config->nSegments, sizeof(config->nSegments));

    for (NnSize bufferIndex = 0; bufferIndex < config->nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config->buffers[bufferIndex];
        network->write(socketIndex, &bufferConfig->size, sizeof(bufferConfig->size));
        writeString(network, socketIndex, bufferConfig->name);
    }

    for (NnSize segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config->segments[segmentIndex];
        network->write(socketIndex, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->write(socketIndex, &segmentConfig->nOps, sizeof(segmentConfig->nOps));
        network->write(socketIndex, &segmentConfig->syncPointers, sizeof(segmentConfig->syncPointers));

        for (NnSize syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
            NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
            network->write(socketIndex, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
            network->write(socketIndex, &syncConfig->syncType, sizeof(syncConfig->syncType));
        }
        for (NnSize opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            network->write(socketIndex, &opConfig->code, sizeof(opConfig->code));
            network->write(socketIndex, &opConfig->index, sizeof(opConfig->index));
            network->write(socketIndex, &opConfig->weightSize, sizeof(opConfig->weightSize));
            network->write(socketIndex, &opConfig->configSize, sizeof(opConfig->configSize));
            writeString(network, socketIndex, opConfig->name);
            network->write(socketIndex, &opConfig->input, sizeof(opConfig->input));
            network->write(socketIndex, &opConfig->output, sizeof(opConfig->output));
            if (opConfig->configSize > 0)
                network->write(socketIndex, opConfig->config, opConfig->configSize);
        }
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs) {
    for (NnSize nodeIndex = 1; nodeIndex < netConfig->nNodes; nodeIndex++) {
        NnSize socketIndex = nodeIndex - 1;
        writeNet(socketIndex, netConfig);
        writeNode(socketIndex, &nodeConfigs[nodeIndex]);
    }
}

NnWorkerConfigReader::NnWorkerConfigReader(NnNetwork *network) {
    this->network = network;
}

NnNetConfig NnWorkerConfigReader::readNet() {
    network->readAck(ROOT_SOCKET_INDEX);
    NnNetConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nBatches, sizeof(config.nBatches));
    network->read(ROOT_SOCKET_INDEX, &config.nNodes, sizeof(config.nNodes));
    network->read(ROOT_SOCKET_INDEX, &config.nPipes, sizeof(config.nPipes));
    config.pipes = new NnPipeConfig[config.nPipes];
    for (NnSize pipeIndex = 0; pipeIndex < config.nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config.pipes[pipeIndex];
        network->read(ROOT_SOCKET_INDEX, &pipeConfig->size, sizeof(pipeConfig->size));
        pipeConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnNodeConfig NnWorkerConfigReader::readNode() {
    network->readAck(ROOT_SOCKET_INDEX);

    NnNodeConfig config;
    network->read(ROOT_SOCKET_INDEX, &config.nodeIndex, sizeof(config.nodeIndex));
    network->read(ROOT_SOCKET_INDEX, &config.nBuffers, sizeof(config.nBuffers));
    network->read(ROOT_SOCKET_INDEX, &config.nSegments, sizeof(config.nSegments));

    config.buffers = new NnBufferConfig[config.nBuffers];
    config.segments = new NnSegmentConfig[config.nSegments];

    for (NnSize bufferIndex = 0; bufferIndex < config.nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config.buffers[bufferIndex];
        network->read(ROOT_SOCKET_INDEX, &bufferConfig->size, sizeof(bufferConfig->size));
        bufferConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }

    for (NnSize segmentIndex = 0; segmentIndex < config.nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config.segments[segmentIndex];
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nOps, sizeof(segmentConfig->nOps));
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->syncPointers, sizeof(segmentConfig->syncPointers));

        if (segmentConfig->nSyncs > 0) {
            segmentConfig->syncs = new NnSyncConfig[segmentConfig->nSyncs];

            for (NnSize syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
                NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
                network->read(ROOT_SOCKET_INDEX, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->syncType, sizeof(syncConfig->syncType));
            }
        }

        if (segmentConfig->nOps > 0) {
            segmentConfig->ops = new NnOpConfig[segmentConfig->nOps];

            for (NnSize opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
                NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
                network->read(ROOT_SOCKET_INDEX, &opConfig->code, sizeof(opConfig->code));
                network->read(ROOT_SOCKET_INDEX, &opConfig->index, sizeof(opConfig->index));
                network->read(ROOT_SOCKET_INDEX, &opConfig->weightSize, sizeof(opConfig->weightSize));
                network->read(ROOT_SOCKET_INDEX, &opConfig->configSize, sizeof(opConfig->configSize));
                opConfig->name = readString(network, ROOT_SOCKET_INDEX);
                network->read(ROOT_SOCKET_INDEX, &opConfig->input, sizeof(opConfig->input));
                network->read(ROOT_SOCKET_INDEX, &opConfig->output, sizeof(opConfig->output));
                if (opConfig->configSize > 0) {
                    opConfig->config = new NnByte[opConfig->configSize];
                    network->read(ROOT_SOCKET_INDEX, opConfig->config, opConfig->configSize);
                }
            }
        }
    }
    network->writeAck(ROOT_SOCKET_INDEX);
    return config;
}

NnRootWeightLoader::NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnSize nNodes) {
    this->executor = executor;
    this->network = network;
    this->nNodes = nNodes;
    this->tempSize = 0;
}

NnRootWeightLoader::~NnRootWeightLoader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnRootWeightLoader::finish() {
    NnSize zeroSize = 0;
    for (NnSize socketIndex = 0; socketIndex < nNodes - 1; socketIndex++) {
        network->write(socketIndex, &zeroSize, sizeof(zeroSize));
        network->readAck(socketIndex);
    }
    if (tempSize > 0) {
        delete[] temp;
        tempSize = 0;
    }
}

void NnRootWeightLoader::allocate(NnSize size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnRootWeightLoader::writeWeight(NnSize nodeIndex, const char *opName, NnSize opIndex, NnSize nBytes, NnByte *weight) {
    NnSize nameSize = std::strlen(opName) + 1;
    NnSize socketIndex = nodeIndex - 1;
    network->write(socketIndex, &nameSize, sizeof(nameSize));
    network->write(socketIndex, opName, nameSize);
    network->write(socketIndex, &opIndex, sizeof(opIndex));
    network->write(socketIndex, &nBytes, sizeof(nBytes));
    network->write(socketIndex, weight, nBytes);
}

NnSize NnRootWeightLoader::loadRoot(const char *opName, NnSize opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, nBytes, weight);
    return nBytes;
}

NnSize NnRootWeightLoader::loadAll(const char *opName, NnSize opIndex, NnSize nBytes, NnByte *weight) {
    for (NnSize nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        if (nodeIndex == 0)
            executor->loadWeight(opName, opIndex, nBytes, weight);
        else
            writeWeight(nodeIndex, opName, opIndex, nBytes, weight);
    }
    return nBytes;
}

NnSize NnRootWeightLoader::loadRowMatmulSlices(const char *opName, NnSize opIndex, NnRowMatmulSlice *slice, NnByte *weight) {
    allocate(slice->sliceSize.nBytes);
    for (NnSize nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        splitRowMatmulWeight(slice, nodeIndex, weight, temp);
        if (nodeIndex == 0)
            executor->loadWeight(opName, opIndex, slice->sliceSize.nBytes, temp);
        else
            writeWeight(nodeIndex, opName, opIndex, slice->sliceSize.nBytes, temp);
    }
    return slice->size.nBytes;
}

NnSize NnRootWeightLoader::loadColMatmulSlices(const char *opName, NnSize opIndex, NnColMatmulSlice *slice, NnByte *weight) {
    allocate(slice->sliceSize.nBytes);
    for (NnSize nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
        splitColMatmulWeight(slice, nodeIndex, weight, temp);
        if (nodeIndex == 0)
            executor->loadWeight(opName, opIndex, slice->sliceSize.nBytes, temp);
        else
            writeWeight(nodeIndex, opName, opIndex, slice->sliceSize.nBytes, temp);
    }
    return slice->size.nBytes;
}

NnWorkerWeightReader::NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network) {
    this->executor = executor;
    this->network = network;
    this->tempSize = 0;
}

NnWorkerWeightReader::~NnWorkerWeightReader() {
    if (tempSize > 0)
        delete[] temp;
}

void NnWorkerWeightReader::allocate(NnSize size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnWorkerWeightReader::read() {
    NnSize nameSize;
    NnSize opIndex;
    NnSize nBytes;
    while (true) {
        network->read(0, &nameSize, sizeof(nameSize));
        if (nameSize == 0) {
            network->writeAck(ROOT_SOCKET_INDEX);
            if (tempSize > 0) {
                delete temp;
                tempSize = 0;
            }
            break;
        }
        char *opName = new char[nameSize];
        network->read(ROOT_SOCKET_INDEX, opName, nameSize);
        network->read(ROOT_SOCKET_INDEX, &opIndex, sizeof(opIndex));
        network->read(ROOT_SOCKET_INDEX, &nBytes, sizeof(nBytes));
        allocate(nBytes);
        network->read(0, temp, nBytes);
        executor->loadWeight(opName, opIndex, nBytes, temp);
        printf("💿 Loaded %22s %3d, %12d kB\n", opName, opIndex, nBytes / 1024);
        delete[] opName;
    }
    printf("💿 Weights loaded\n");
}
