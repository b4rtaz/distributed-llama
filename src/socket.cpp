#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "socket.hpp"
#include <stdexcept>

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

static inline void setNotBlocking(int socket, bool enabled) {
    int flags = fcntl(socket, F_GETFL, 0);
    if (enabled) {
        flags |= O_NONBLOCK;
    } else {
        flags = flags & (~O_NONBLOCK);
    }
    if (fcntl(socket, F_SETFL, flags) < 0)
        throw std::runtime_error("Error setting socket to non-blocking");
}

static inline void setNoDelay(int socket) {
    int flag = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error setting socket to no-delay");
}

static inline void disableReuse(int socket) {
    int flag = 0;
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error disabling SO_REUSEADDR");
    if (setsockopt(socket, SOL_SOCKET, SO_REUSEPORT, (char*)&flag, sizeof(int)) < 0)
        throw std::runtime_error("Error disabling SO_REUSEPORT");
}

static inline void writeSocket(int socket, const char* data, size_t size) {
    while (size > 0) {
        int s = send(socket, (char*)data, size, 0);
        if (s < 0) {
            if (SOCKET_LAST_ERRCODE == EAGAIN) {
                continue;
            }
            throw WriteSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
        } else if (s == 0) {
            throw WriteSocketException(0, "Socket closed");
        }
        size -= s;
        data = (char*)data + s;
    }
}

static inline void readSocket(int socket, char* data, size_t size) {
    while (size > 0) {
        int r = recv(socket, (char*)data, size, 0);
        if (r < 0) {
            if (SOCKET_LAST_ERRCODE == EAGAIN) {
                continue;
            }
            throw ReadSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
        } else if (r == 0) {
            throw ReadSocketException(0, "Socket closed");
        }
        data = (char*)data + r;
        size -= r;
    }
}

ReadSocketException::ReadSocketException(int code, const char* message) {
    this->code = code;
    this->message = message;
}

WriteSocketException::WriteSocketException(int code, const char* message) {
    this->code = code;
    this->message = message;
}

SocketPool* SocketPool::connect(unsigned int nSockets, char** hosts, int* ports) {
    int* sockets = new int[nSockets];
    struct sockaddr_in addr;

    for (unsigned int i = 0; i < nSockets; i++) {
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(hosts[i]);
        addr.sin_port = htons(ports[i]);

        int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (clientSocket < 0)
            throw std::runtime_error("Cannot create socket");

        int connectResult = ::connect(clientSocket, (struct sockaddr*)&addr, sizeof(addr));
        if (connectResult != 0) {
            printf("Cannot connect to %s:%d (%s)\n", hosts[i], ports[i], SOCKET_LAST_ERROR);
            throw std::runtime_error("Cannot connect");
        }

        setNoDelay(clientSocket);
        sockets[i] = clientSocket;
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
    }
}

void SocketPool::setNotBlocking(bool enabled) {
    for (unsigned int i = 0; i < nSockets; i++) {
        ::setNotBlocking(sockets[i], enabled);
    }
}

void SocketPool::write(unsigned int socketIndex, const char* data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    sentBytes += size;
    writeSocket(sockets[socketIndex], data, size);
}

void SocketPool::read(unsigned int socketIndex, char* data, size_t size) {
    assert(socketIndex >= 0 && socketIndex < nSockets);
    recvBytes += size;
    readSocket(sockets[socketIndex], data, size);
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
                ssize_t s = send(socket, io->data, io->size, 0);
                if (s < 0) {
                    if (SOCKET_LAST_ERRCODE == EAGAIN) {
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
                    if (SOCKET_LAST_ERRCODE == EAGAIN) {
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

Socket Socket::accept(int port) {
    const char* host = "0.0.0.0";
    int serverSocket;
    struct sockaddr_in serverAddr;
    struct sockaddr_in clientAddr;

    serverSocket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0)
        throw std::runtime_error("Cannot create socket");

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        printf("Cannot bind %s:%d\n", host, port);
        throw std::runtime_error("Cannot bind port");
    }

    int listenResult = listen(serverSocket, 1);
    if (listenResult != 0) {
        printf("Cannot listen %s:%d\n", host, port);
        exit(EXIT_FAILURE);
    }
    printf("Listening on %s:%d...\n", host, port);

    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0)
        throw std::runtime_error("Error accepting connection");

    disableReuse(serverSocket);
    shutdown(serverSocket, 2);
    close(serverSocket);
    setNoDelay(clientSocket);

    printf("Client connected\n");
    return Socket(clientSocket);
}

Socket::Socket(int socket) {
    this->socket = socket;
}

Socket::~Socket() {
    shutdown(socket, 2);
    close(socket);
}

void Socket::setNotBlocking(bool enabled) {
    ::setNotBlocking(socket, enabled);
}

void Socket::write(const char* data, size_t size) {
    writeSocket(socket, data, size);
}

void Socket::read(char* data, size_t size) {
    readSocket(socket, data, size);
}
