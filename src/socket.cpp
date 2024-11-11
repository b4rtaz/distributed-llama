#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <ctime>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>
#include "socket.hpp"

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

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

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

static inline void setReuseAddr(int socket) {
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

static inline void writeSocket(int socket, const void* data, size_t size) {
    while (size > 0) {
        int s = send(socket, (const char*)data, size, 0);
        if (s < 0) {
            if (isEagainError()) {
                continue;
            }
            throw WriteSocketException(0, "Error writing to socket");
        } else if (s == 0) {
            throw WriteSocketException(0, "Socket closed");
        }
        size -= s;
        data = (const char*)data + s;
    }
}

static inline bool tryReadSocket(int socket, void* data, size_t size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    size_t s = size;
    while (s > 0) {
        int r = recv(socket, (char*)data, s, 0);
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
            throw ReadSocketException(0, "Error reading from socket");
        } else if (r == 0) {
            throw ReadSocketException(0, "Socket closed");
        }
        data = (char*)data + r;
        s -= r;
    }
    return true;
}

static inline void readSocket(int socket, void* data, size_t size) {
    if (!tryReadSocket(socket, data, size, 0)) {
        throw std::runtime_error("Error reading from socket");
    }
}

static inline int connectSocket(char* host, int port) {
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

static inline int acceptSocket(int serverSocket) {
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

ReadSocketException::ReadSocketException(int code, const char* message) {
    this->code = code;
    this->message = message;
}

WriteSocketException::WriteSocketException(int code, const char* message) {
    this->code = code;
    this->message = message;
}

SocketPool* SocketPool::serve(int port) {
    const char* host = "0.0.0.0";
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

    unsigned int nSockets;
    unsigned int nodeIndex;
    int rootSocket = acceptSocket(serverSocket);

    readSocket(rootSocket, &nSockets, sizeof(nSockets));
    readSocket(rootSocket, &nodeIndex, sizeof(nodeIndex));

    unsigned int nNodes = nSockets - 1; // nSockets - 1 root node
    int* sockets = new int[nSockets];
    sockets[0] = rootSocket;
    char** hosts = new char*[nNodes];
    int* ports = new int[nNodes];

    printf("The root node has connected\n");
    printf("⭕ nodeIndex: %d\n", nodeIndex);
    printf("⭕ socket[0]: root node\n");

    size_t hostLen;
    for (unsigned int i = 0; i < nNodes; i++) {
        readSocket(rootSocket, &hostLen, sizeof(hostLen));
        hosts[i] = new char[hostLen];
        readSocket(rootSocket, hosts[i], hostLen);
        readSocket(rootSocket, &ports[i], sizeof(ports[i]));
    }

    for (unsigned int i = 0; i < nNodes; i++) {
        unsigned int socketIndex = i + 1;
        if (i >= nodeIndex) {
            printf("⭕ socket[%d]: connected to %s:%d\n", socketIndex, hosts[i], ports[i]);
            sockets[socketIndex] = connectSocket(hosts[i], ports[i]);
        } else {
            printf("⭕ socket[%d]: accepted from %s:%d\n", socketIndex, hosts[i], ports[i]);
            sockets[socketIndex] = acceptSocket(serverSocket);
        }
    }

    unsigned int confirm = 0x555;
    writeSocket(rootSocket, &confirm, sizeof(confirm));

    for (unsigned int i = 0; i < nNodes; i++)
        delete[] hosts[i];
    delete[] hosts;
    delete[] ports;

    shutdown(serverSocket, 2);
    close(serverSocket);
    return new SocketPool(nSockets, sockets);
}

SocketPool* SocketPool::connect(unsigned int nSockets, char** hosts, int* ports) {
    int* sockets = new int[nSockets];
    struct sockaddr_in addr;
    for (unsigned int i = 0; i < nSockets; i++) {
        int socket = connectSocket(hosts[i], ports[i]);
        sockets[i] = socket;
        writeSocket(socket, &nSockets, sizeof(nSockets));
        writeSocket(socket, &i, sizeof(i)); // send node index
        for (unsigned int j = 0; j < nSockets; j++) {
            if (j == i)
                continue;
            size_t hostLen = strlen(hosts[j]) + 1;
            writeSocket(socket, &hostLen, sizeof(hostLen));
            writeSocket(socket, hosts[j], hostLen);
            writeSocket(socket, &ports[j], sizeof(ports[j]));
        }
    }
    unsigned int confirm;
    for (unsigned int i = 0; i < nSockets; i++) {
        readSocket(sockets[i], &confirm, sizeof(confirm));
        if (confirm != 0x555) {
            throw std::runtime_error("Cannot create network");
        }
        printf("⭕ socket[%d]: connected to %s:%d\n", i, hosts[i], ports[i]);
    }
    return new SocketPool(nSockets, sockets);
}

SocketPool::SocketPool(unsigned int nSockets, int* sockets) {
    this->nSockets = nSockets;
    this->sockets = sockets;
    this->sentBytes.exchange(0);
    this->recvBytes.exchange(0);
}

SocketPool::~SocketPool() {
    for (unsigned int i = 0; i < nSockets; i++) {
        shutdown(sockets[i], 2);
        close(sockets[i]);
    }
    delete[] sockets;
}

void SocketPool::setTurbo(bool enabled) {
    for (unsigned int i = 0; i < nSockets; i++) {
        ::setNonBlocking(sockets[i], enabled);
    }
}

void SocketPool::write(unsigned int socketIndex, const void* data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    sentBytes += size;
    writeSocket(sockets[socketIndex], data, size);
}

void SocketPool::read(unsigned int socketIndex, void* data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    recvBytes += size;
    readSocket(sockets[socketIndex], data, size);
}

bool SocketPool::tryRead(unsigned int socketIndex, void* data, size_t size, unsigned long maxAttempts) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    recvBytes += size;
    return tryReadSocket(sockets[socketIndex], data, size, maxAttempts);
}

void SocketPool::writeMany(unsigned int n, SocketIo* ios) {
    bool isWriting;
    for (unsigned int i = 0; i < n; i++) {
        SocketIo* io = &ios[i];
        assert(io->socketIndex >= 0 && io->socketIndex < nSockets);
        sentBytes += io->size;
    }
    do {
        isWriting = false;
        for (unsigned int i = 0; i < n; i++) {
            SocketIo* io = &ios[i];
            if (io->size > 0) {
                isWriting = true;
                int socket = sockets[io->socketIndex];
                ssize_t s = send(socket, (const char*)io->data, io->size, 0);
                if (s < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw WriteSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (s == 0) {
                    throw WriteSocketException(0, "Socket closed");
                }
                io->size -= s;
                io->data = (char*)io->data + s;
            }
        }
    } while (isWriting);
}

void SocketPool::readMany(unsigned int n, SocketIo* ios) {
    bool isReading;
    for (unsigned int i = 0; i < n; i++) {
        SocketIo* io = &ios[i];
        assert(io->socketIndex >= 0 && io->socketIndex < nSockets);
        recvBytes += io->size;
    }
    do {
        isReading = false;
        for (unsigned int i = 0; i < n; i++) {
            SocketIo* io = &ios[i];
            if (io->size > 0) {
                isReading = true;
                int socket = sockets[io->socketIndex];
                ssize_t r = recv(socket, (char*)io->data, io->size, 0);
                if (r < 0) {
                    if (isEagainError()) {
                        continue;
                    }
                    throw ReadSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
                } else if (r == 0) {
                    throw ReadSocketException(0, "Socket closed");
                }
                io->size -= r;
                io->data = (char*)io->data + r;
            }
        }
    } while (isReading);
}

void SocketPool::getStats(size_t* sentBytes, size_t* recvBytes) {
    *sentBytes = this->sentBytes;
    *recvBytes = this->recvBytes;
    this->sentBytes.exchange(0);
    this->recvBytes.exchange(0);
}
