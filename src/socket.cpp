#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <arpa/inet.h>
#include <errno.h>
#include <string.h>
#include "socket.hpp"

#define SOCKET_LAST_ERRCODE errno
#define SOCKET_LAST_ERROR strerror(errno)

SocketPool SocketPool::connect(unsigned int nSockets, char** hosts, int* ports) {
    int* sockets = new int[nSockets];
    struct sockaddr_in addr;

    for (unsigned int i = 0; i < nSockets; i++) {
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = inet_addr(hosts[i]);
        addr.sin_port = htons(ports[i]);

        int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (clientSocket < 0) {
            printf("Error creating socket\n");
            exit(EXIT_FAILURE);
        }

        int connectResult = ::connect(clientSocket, (struct sockaddr*)&addr, sizeof(addr));
        if (connectResult != 0) {
            printf("Cannot connect to %s:%d (%s)\n", hosts[i], ports[i], SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }

        sockets[i] = clientSocket;
    }
    return SocketPool(nSockets, sockets);
}

SocketPool::SocketPool(unsigned int nSockets, int* sockets) {
    this->nSockets = nSockets;
    this->sockets = sockets;
}

void SocketPool::write(unsigned int socketIndex, const char* data, size_t size) {
    assert(socketIndex < nSockets);
    int socket = sockets[socketIndex];
    while (size > 0) {
        int s = send(socket, (char*)data, size, 0);
        if (s <= 0) {
            printf("Error sending data %d (%s)\n", SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        size -= s;
        data = (char*)data + s;
    }
}

void SocketPool::read(unsigned int socketIndex, char* data, size_t size) {
}

Socket Socket::accept(int port) {
    const char* host = "0.0.0.0";
    int serverSocket;
    struct sockaddr_in serverAddr;
    struct sockaddr_in clientAddr;

    serverSocket = ::socket(AF_INET, SOCK_STREAM, 0);
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

    int listenResult = listen(serverSocket, 1);
    if (listenResult != 0) {
        printf("Cannot listen %s:%d\n", host, port);
        exit(EXIT_FAILURE);
    }
    printf("Listening on %s:%d...\n", host, port);

    socklen_t clientAddrSize = sizeof(clientAddr);
    int clientSocket = ::accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
    if (clientSocket < 0) {
        printf("Error accepting connection\n");
        exit(EXIT_FAILURE);
    }

    printf("Client connected\n");

    shutdown(serverSocket, 2);
    return Socket(clientSocket);
}

Socket::Socket(int socket) {
    this->socket = socket;
}

Socket::~Socket() {
    shutdown(socket, 2);
}

void Socket::write(const char* data, size_t size) {
}

void Socket::read(char* data, size_t size) {
    while (size > 0) {
        int r = recv(socket, (char*)data, size, 0);
        if (r <= 0) {
            printf("Error receiving data %d (%s)\n", SOCKET_LAST_ERRCODE, SOCKET_LAST_ERROR);
            exit(EXIT_FAILURE);
        }
        data = (char*)data + r;
        size -= r;
    }
}
