#ifndef SOCKET_HPP
#define SOCKET_HPP

#include <cstddef>

class SocketPool {
private:
    int* sockets;

public:
    static SocketPool connect(unsigned int nSockets, char** hosts, int* ports);

    unsigned int nSockets;

    SocketPool(unsigned int nSockets, int* sockets);
    ~SocketPool();

    void enableTurbo();
    void write(unsigned int socketIndex, const char* data, size_t size);
    void read(unsigned int socketIndex, char* data, size_t size);
};

class Socket {
private:
    int socket;

public:
    static Socket accept(int port);

    Socket(int socket);
    ~Socket();

    void enableTurbo();
    void write(const char* data, size_t size);
    void read(char* data, size_t size);
};

#endif
