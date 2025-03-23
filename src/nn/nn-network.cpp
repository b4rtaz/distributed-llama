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

#define ACK 23571114
#define MAX_CHUNK_SIZE 4096

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

void writeSocket(int socket, const void *data, NnSize size) {
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

static inline bool tryReadSocket(int socket, void *data, NnSize size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    NnSize s = size;
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

void readSocket(int socket, void *data, NnSize size) {
    if (!tryReadSocket(socket, data, size, 0)) {
        throw std::runtime_error("Error reading from socket");
    }
}

static void readAckPacket(int socket) {
    NnUint packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != ACK)
        throw std::runtime_error("Invalid ack packet");
}

static void writeAckPacket(int socket) {
    NnUint packet = ACK;
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

    NnUint nSockets;
    NnUint nodeIndex;
    int rootSocket = acceptSocket(serverSocket);
    printf("⭕ The root node has connected\n");

    readSocket(rootSocket, &nSockets, sizeof(nSockets));
    NnUint nNodes = nSockets - 1; // nSockets - 1 root node
    printf("⭕ nNodes: %d\n", nNodes);
    readSocket(rootSocket, &nodeIndex, sizeof(nodeIndex));
    printf("⭕ NodeIndex: %d\n", nodeIndex);

    int *sockets = new int[nSockets];
    sockets[0] = rootSocket;
    char* *hosts = new char*[nNodes];
    int *ports = new int[nNodes];
    printf("⭕ Socket[0]: accepted root node\n");

    NnUint hostLen;
    for (NnUint i = 0; i < nNodes; i++) {
        readSocket(rootSocket, &hostLen, sizeof(hostLen));
        hosts[i] = new char[hostLen];
        readSocket(rootSocket, hosts[i], hostLen);
        readSocket(rootSocket, &ports[i], sizeof(ports[i]));
    }

    writeAckPacket(rootSocket);

    // We need to wait here until the root node will send a "root is ready" packet
    readAckPacket(rootSocket);

    for (NnUint i = 0; i < nNodes; i++) {
        NnUint socketIndex = i + 1;
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

    for (NnUint i = 0; i < nNodes; i++)
        delete[] hosts[i];
    delete[] hosts;
    delete[] ports;

    shutdown(serverSocket, 2);
    close(serverSocket);
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(nSockets, sockets));
}

std::unique_ptr<NnNetwork> NnNetwork::connect(NnUint nSockets, char **hosts, NnUint *ports) {
    assert(nSockets > 0);

    int *sockets = new int[nSockets];
    struct sockaddr_in addr;
    for (NnUint i = 0; i < nSockets; i++) {
        printf("⭕ Socket[%d]: connecting to %s:%d worker\n", i, hosts[i], ports[i]);
        int socket = connectSocket(hosts[i], ports[i]);
        sockets[i] = socket;
        writeSocket(socket, &nSockets, sizeof(nSockets));
        writeSocket(socket, &i, sizeof(i)); // send node index
        for (NnUint j = 0; j < nSockets; j++) {
            if (j == i)
                continue;
            NnUint hostLen = strlen(hosts[j]) + 1;
            writeSocket(socket, &hostLen, sizeof(hostLen));
            writeSocket(socket, hosts[j], hostLen);
            writeSocket(socket, &ports[j], sizeof(ports[j]));
        }
        readAckPacket(socket);
        printf("⭕ Socket[%d]: connected\n", i);
    }
    for (NnUint i = 0; i < nSockets; i++) {
        writeAckPacket(sockets[i]);
    }
    printf("⭕ Network is initialized\n");
    return std::unique_ptr<NnNetwork>(new NnNetwork(nSockets, sockets));
}

NnNetwork::NnNetwork(NnUint nSockets, int *sockets) {
    this->nSockets = nSockets;
    this->sockets = sockets;
    this->sentBytes = new NnSize[nSockets];
    this->recvBytes = new NnSize[nSockets];
}

NnNetwork::~NnNetwork() {
    delete[] sentBytes;
    delete[] recvBytes;
    for (NnUint i = 0; i < nSockets; i++) {
        shutdown(sockets[i], 2);
        close(sockets[i]);
    }
    delete[] sockets;
    printf("⭕ Network is closed\n");
}

void NnNetwork::setTurbo(bool enabled) {
    for (NnUint i = 0; i < nSockets; i++) {
        ::setNonBlocking(sockets[i], enabled);
    }
}

void NnNetwork::write(const NnUint socketIndex, const void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        writeSocket(s, current, chunkSize);
        current += chunkSize;
    }
    sentBytes[socketIndex] += size;
}

void NnNetwork::read(const NnUint socketIndex, void *data, const NnSize size) {
    assert(socketIndex < nSockets);

    NnByte *current = (NnByte *)data;
    int s = sockets[socketIndex];
    for (NnSize chunk = 0; chunk < size; chunk += MAX_CHUNK_SIZE) {
        NnSize chunkSize = chunk + MAX_CHUNK_SIZE < size ? MAX_CHUNK_SIZE : size - chunk;
        readSocket(s, current, chunkSize);
        current += chunkSize;
    }
    recvBytes[socketIndex] += size;
}

void NnNetwork::writeAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    writeAckPacket(sockets[socketIndex]);
}

void NnNetwork::readAck(const NnUint socketIndex) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    readAckPacket(sockets[socketIndex]);
}

bool NnNetwork::tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    if (tryReadSocket(sockets[socketIndex], data, size, maxAttempts)) {
        recvBytes[socketIndex] += size;
        return true;
    }
    return false;
}

void NnNetwork::writeMany(NnUint n, NnSocketIo *ios) {
    bool isWriting;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        sentBytes[io->socketIndex] += io->size;
    }
    do {
        isWriting = false;
        for (NnUint i = 0; i < n; i++) {
            NnSocketIo *io = &ios[i];
            if (io->size > 0) {
                isWriting = true;
                int socket = sockets[io->socketIndex];
                ssize_t chunkSize = io->size > MAX_CHUNK_SIZE ? MAX_CHUNK_SIZE : io->size;
                ssize_t s = send(socket, (const char*)io->data, chunkSize, 0);
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
}

void NnNetwork::writeAll(void *data, NnSize size) {
    std::vector<NnSocketIo> ios(nSockets);
    for (NnUint i = 0; i < nSockets; i++) {
        NnSocketIo *io = &ios[i];
        io->socketIndex = i;
        io->data = data;
        io->size = size;
    }
    writeMany(nSockets, &ios[0]);
}

void NnNetwork::readMany(NnUint n, NnSocketIo *ios) {
    bool isReading;
    NnSize nBytes = 0;
    for (NnUint i = 0; i < n; i++) {
        NnSocketIo *io = &ios[i];
        assert(io->socketIndex < nSockets);
        recvBytes[io->socketIndex] += io->size;
    }
    do {
        isReading = false;
        for (NnUint i = 0; i < n; i++) {
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
}

void NnNetwork::getStats(NnSize *sentBytes, NnSize *recvBytes) {
    *sentBytes = 0;
    *recvBytes = 0;
    for (NnUint i = 0; i < nSockets; i++) {
        *sentBytes += this->sentBytes[i];
        *recvBytes += this->recvBytes[i];
    }
    resetStats();
}

void NnNetwork::resetStats() {
    for (NnUint i = 0; i < nSockets; i++) {
        sentBytes[i] = 0;
        recvBytes[i] = 0;
    }
}

static void syncWithRoot(NnNetwork *network, NnByte nodeIndex, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    if (nodeIndex == 0) {
        // root

        NnUint nSocketsPerThread = network->nSockets / nThreads + (network->nSockets % nThreads > threadIndex ? 1 : 0);
        if (nSocketsPerThread == 0) return;

        std::vector<NnSocketIo> ios(nSocketsPerThread);
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
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

static void syncNodeSlices(bool onlyFromWorkerToRoot, NnNetwork *network, NnUint nodeIndex, NnUint nNodes, NnByte *buffer, NnSize nBytes, NnUint nThreads, NnUint threadIndex) {
    bool isWorker = nodeIndex != 0;
    NnUint nSockets = onlyFromWorkerToRoot && isWorker ? 1 : network->nSockets;
    NnUint nSocketsPerThread = nSockets / nThreads + (nSockets % nThreads > threadIndex ? 1 : 0);
    if (nSocketsPerThread == 0) return;
    NnSize sliceBytes = nBytes / nNodes;

    std::vector<NnSocketIo> ios(nSocketsPerThread);

    if (!onlyFromWorkerToRoot || isWorker) {
        NnByte *mySliceData = &buffer[sliceBytes * nodeIndex];

        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint socketIndex = threadIndex + i * nThreads;
            ios[i].socketIndex = socketIndex;
            ios[i].data = mySliceData;
            ios[i].size = sliceBytes;
        }
        network->writeMany(nSocketsPerThread, &ios[0]);
    }

    if (!onlyFromWorkerToRoot || !isWorker) {
        for (NnUint i = 0; i < nSocketsPerThread; i++) {
            NnUint socketIndex = threadIndex + i * nThreads;
            NnUint sliceIndex = socketIndex >= nodeIndex ? socketIndex + 1 : socketIndex;
            NnByte *sliceData = &buffer[sliceBytes * sliceIndex];
            ios[i].socketIndex = socketIndex;
            ios[i].data = sliceData;
            ios[i].size = sliceBytes;
        }
        network->readMany(nSocketsPerThread, &ios[0]);
    }
}

NnNetworkNodeSynchronizer::NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    this->network = network;
    this->execution = execution;
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
}

void NnNetworkNodeSynchronizer::sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];

    for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
        NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
        NnByte *pipe = execution->pipes[syncConfig->pipeIndex];
        NnPipeConfig *pipeConfig = &netConfig->pipes[syncConfig->pipeIndex];
        NnSize batchBytes = getBytes(pipeConfig->size.floatType, pipeConfig->size.x);

        for (NnUint batchIndex = 0; batchIndex < execution->batchSize; batchIndex++) {
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

static void writeString(NnNetwork *network, NnUint socketIndex, char *str) {
    NnUint bytes = std::strlen(str) + 1;
    network->write(socketIndex, &bytes, sizeof(NnUint));
    network->write(socketIndex, str, bytes);
}

static char *readString(NnNetwork *network, NnUint socketIndex) {
    NnUint bytes;
    network->read(socketIndex, &bytes, sizeof(NnUint));
    char *str = new char[bytes];
    network->read(socketIndex, str, bytes);
    return str;
}

NnRootConfigWriter::NnRootConfigWriter(NnNetwork *network) {
    this->network = network;
}

void NnRootConfigWriter::writeNet(NnUint socketIndex, NnNetConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nBatches, sizeof(config->nBatches));
    network->write(socketIndex, &config->nNodes, sizeof(config->nNodes));
    network->write(socketIndex, &config->nPipes, sizeof(config->nPipes));
    for (NnUint pipeIndex = 0; pipeIndex < config->nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config->pipes[pipeIndex];
        network->write(socketIndex, &pipeConfig->size, sizeof(pipeConfig->size));
        writeString(network, socketIndex, pipeConfig->name);
    }
    network->write(socketIndex, &config->nPreSyncs, sizeof(config->nPreSyncs));
    for (NnUint preSyncIndex = 0; preSyncIndex < config->nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config->preSyncs[preSyncIndex];
        network->write(socketIndex, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
    }
    network->readAck(socketIndex);
}

void NnRootConfigWriter::writeNode(NnUint socketIndex, NnNodeConfig *config) {
    network->writeAck(socketIndex);
    network->write(socketIndex, &config->nodeIndex, sizeof(config->nodeIndex));
    network->write(socketIndex, &config->nBuffers, sizeof(config->nBuffers));
    network->write(socketIndex, &config->nSegments, sizeof(config->nSegments));

    for (NnUint bufferIndex = 0; bufferIndex < config->nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config->buffers[bufferIndex];
        network->write(socketIndex, &bufferConfig->size, sizeof(bufferConfig->size));
        writeString(network, socketIndex, bufferConfig->name);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config->segments[segmentIndex];
        network->write(socketIndex, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->write(socketIndex, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
            NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
            network->write(socketIndex, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
            network->write(socketIndex, &syncConfig->syncType, sizeof(syncConfig->syncType));
        }
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
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
    for (NnUint nodeIndex = 1; nodeIndex < netConfig->nNodes; nodeIndex++) {
        NnUint socketIndex = nodeIndex - 1;
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
    for (NnUint pipeIndex = 0; pipeIndex < config.nPipes; pipeIndex++) {
        NnPipeConfig *pipeConfig = &config.pipes[pipeIndex];
        network->read(ROOT_SOCKET_INDEX, &pipeConfig->size, sizeof(pipeConfig->size));
        pipeConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }
    network->read(ROOT_SOCKET_INDEX, &config.nPreSyncs, sizeof(config.nPreSyncs));
    config.preSyncs = new NnPreSyncConfig[config.nPreSyncs];
    for (NnUint preSyncIndex = 0; preSyncIndex < config.nPreSyncs; preSyncIndex++) {
        NnPreSyncConfig *preSyncConfig = &config.preSyncs[preSyncIndex];
        network->read(ROOT_SOCKET_INDEX, &preSyncConfig->pipeIndex, sizeof(preSyncConfig->pipeIndex));
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

    for (NnUint bufferIndex = 0; bufferIndex < config.nBuffers; bufferIndex++) {
        NnBufferConfig *bufferConfig = &config.buffers[bufferIndex];
        network->read(ROOT_SOCKET_INDEX, &bufferConfig->size, sizeof(bufferConfig->size));
        bufferConfig->name = readString(network, ROOT_SOCKET_INDEX);
    }

    for (NnUint segmentIndex = 0; segmentIndex < config.nSegments; segmentIndex++) {
        NnSegmentConfig *segmentConfig = &config.segments[segmentIndex];
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nSyncs, sizeof(segmentConfig->nSyncs));
        network->read(ROOT_SOCKET_INDEX, &segmentConfig->nOps, sizeof(segmentConfig->nOps));

        if (segmentConfig->nSyncs > 0) {
            segmentConfig->syncs = new NnSyncConfig[segmentConfig->nSyncs];

            for (NnUint syncIndex = 0; syncIndex < segmentConfig->nSyncs; syncIndex++) {
                NnSyncConfig *syncConfig = &segmentConfig->syncs[syncIndex];
                network->read(ROOT_SOCKET_INDEX, &syncConfig->pipeIndex, sizeof(syncConfig->pipeIndex));
                network->read(ROOT_SOCKET_INDEX, &syncConfig->syncType, sizeof(syncConfig->syncType));
            }
        }

        if (segmentConfig->nOps > 0) {
            segmentConfig->ops = new NnOpConfig[segmentConfig->nOps];

            for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
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

NnRootWeightLoader::NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes) {
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
    NnUint zeroSize = 0;
    for (NnUint socketIndex = 0; socketIndex < nNodes - 1; socketIndex++) {
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

void NnRootWeightLoader::writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    NnUint nameSize = std::strlen(opName) + 1;
    NnUint socketIndex = nodeIndex - 1;
    network->write(socketIndex, &nameSize, sizeof(nameSize));
    network->write(socketIndex, opName, nameSize);
    network->write(socketIndex, &opIndex, sizeof(opIndex));
    network->write(socketIndex, &nBytes, sizeof(nBytes));
    network->write(socketIndex, weight, nBytes);
}

NnSize NnRootWeightLoader::loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, nBytes, weight);
    return nBytes;
}

NnSize NnRootWeightLoader::loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
    executor->loadWeight(opName, opIndex, nBytes, weight);

    if (nNodes > 1) {
        for (NnUint nodeIndex = 1; nodeIndex < nNodes; nodeIndex++)
            writeWeight(nodeIndex, opName, opIndex, nBytes, weight);
    }
    return nBytes;
}

NnSize NnRootWeightLoader::loadRowMatmulSlices(const char *opName, NnUint opIndex, NnRowMatmulSlice *slice, NnByte *weight) {
    if (nNodes == 1) {
        executor->loadWeight(opName, opIndex, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitRowMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0)
                executor->loadWeight(opName, opIndex, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, slice->sliceSize.nBytes, temp);
        }
    }
    return slice->size.nBytes;
}

NnSize NnRootWeightLoader::loadColMatmulSlices(const char *opName, NnUint opIndex, NnColMatmulSlice *slice, NnByte *weight) {
    if (nNodes == 1) {
        executor->loadWeight(opName, opIndex, slice->sliceSize.nBytes, weight);
    } else {
        allocate(slice->sliceSize.nBytes);
        for (NnUint nodeIndex = 0; nodeIndex < nNodes; nodeIndex++) {
            splitColMatmulWeight(slice, nodeIndex, weight, temp);
            if (nodeIndex == 0)
                executor->loadWeight(opName, opIndex, slice->sliceSize.nBytes, temp);
            else
                writeWeight(nodeIndex, opName, opIndex, slice->sliceSize.nBytes, temp);
        }
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

void NnWorkerWeightReader::allocate(NnUint size) {
    if (tempSize < size) {
        if (tempSize > 0)
            delete[] temp;
        tempSize = size;
        temp = new NnByte[size];
    }
}

void NnWorkerWeightReader::read() {
    NnUint nameSize;
    NnUint opIndex;
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
        std::unique_ptr<char[]> opNamePtr(new char[nameSize]);
        char *opName = opNamePtr.get();
        network->read(ROOT_SOCKET_INDEX, opName, nameSize);
        network->read(ROOT_SOCKET_INDEX, &opIndex, sizeof(opIndex));
        network->read(ROOT_SOCKET_INDEX, &nBytes, sizeof(nBytes));
        allocate(nBytes);
        network->read(0, temp, nBytes);
        executor->loadWeight(opName, opIndex, nBytes, temp);
        printf("💿 Loaded %22s %3d, %12zu kB\n", opName, opIndex, nBytes / 1024);
    }
    printf("💿 Weights loaded\n");
}
