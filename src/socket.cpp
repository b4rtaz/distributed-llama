#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <ctime>
#include <unistd.h>
#include <stdexcept>
#include <vector>
#include "socket.hpp"

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

static inline void setNonBlocking(int socket, bool enabled) {
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

static inline void setQuickAck(int socket) {
#ifdef TCP_QUICKACK
    int value = 1;
    if (setsockopt(socket, IPPROTO_TCP, TCP_QUICKACK, (char*)&value, sizeof(int)) < 0)
        throw std::runtime_error("Error setting quick ack");
#endif
}

static inline void writeSocket(int socket, const void* data, size_t size) {
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

static inline bool tryReadSocket(int socket, void* data, size_t size, unsigned long maxAttempts) {
    // maxAttempts = 0 means infinite attempts
    size_t s = size;
    while (s > 0) {
        int r = recv(socket, data, s, 0);
        if (r < 0) {
            if (SOCKET_LAST_ERRCODE == EAGAIN) {
                if (s == size && maxAttempts > 0) {
                    maxAttempts--;
                    if (maxAttempts == 0) {
                        return false;
                    }
                }
                continue;
            }
            throw ReadSocketException(SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
        } else if (r == 0) {
            throw ReadSocketException(0, "Socket closed");
        }
        data = (char*)data + r;
        s -= r;
    }
    return true;
}

static inline void readSocket(int socket, void* data, size_t size) {
    assert(tryReadSocket(socket, data, size, 0));
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
        setQuickAck(clientSocket);
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

Socket SocketServer::accept() {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(socket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0)
        throw std::runtime_error("Error accepting connection");
    setNoDelay(clientSocket);
    setQuickAck(clientSocket);
    return Socket(clientSocket);
}

Socket::Socket(int socket) {
    this->socket = socket;
}

Socket::~Socket() {
    shutdown(socket, 2);
    close(socket);
}

void Socket::setTurbo(bool enabled) {
    ::setNonBlocking(socket, enabled);
}

void Socket::write(const void* data, size_t size) {
    writeSocket(socket, data, size);
}

void Socket::read(void* data, size_t size) {
    readSocket(socket, data, size);
}

bool Socket::tryRead(void* data, size_t size, unsigned long maxAttempts) {
    return tryReadSocket(socket, data, size, maxAttempts);
}

std::vector<char> Socket::readHttpRequest() {
        std::vector<char> httpRequest;
        char buffer[1024 * 1024]; // TODO: this should be refactored asap
        ssize_t bytesRead;
        
        // Peek into the socket buffer to check available data
        bytesRead = recv(socket, buffer, sizeof(buffer), MSG_PEEK);
        if (bytesRead <= 0) {
            // No data available or error occurred
            if (bytesRead == 0) {
                // No more data to read
                return httpRequest;
            } else {
                // Error while peeking
                throw std::runtime_error("Error while peeking into socket");
            }
        }
        
        // Resize buffer according to the amount of data available
        std::vector<char> peekBuffer(bytesRead);
        bytesRead = recv(socket, peekBuffer.data(), bytesRead, 0);
        if (bytesRead <= 0) {
            // Error while reading
            throw std::runtime_error("Error while reading from socket");
        }

        // Append data to httpRequest
        httpRequest.insert(httpRequest.end(), peekBuffer.begin(), peekBuffer.end());
        
        return httpRequest;
    }

SocketServer::SocketServer(int port) {
    const char* host = "0.0.0.0";
    struct sockaddr_in serverAddr;

    socket = ::socket(AF_INET, SOCK_STREAM, 0);
    if (socket < 0)
        throw std::runtime_error("Cannot create socket");

    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult = bind(socket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        printf("Cannot bind %s:%d\n", host, port);
        throw std::runtime_error("Cannot bind port");
    }

    int listenResult = listen(socket, 1);
    if (listenResult != 0) {
        printf("Cannot listen %s:%d\n", host, port);
        throw std::runtime_error("Cannot listen port");
    }
    printf("Listening on %s:%d...\n", host, port);
}

SocketServer::~SocketServer() {
    shutdown(socket, 2);
    close(socket);
}
