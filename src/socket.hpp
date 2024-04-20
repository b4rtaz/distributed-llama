#ifndef SOCKET_HPP
#define SOCKET_HPP

#include <atomic>
#include <cstddef>
#include <exception>

class ReadSocketException : public std::exception {
public:
    int code;
    const char* message;
    ReadSocketException(int code, const char* message);
};
class WriteSocketException : public std::exception {
public:
    int code;
    const char* message;
    WriteSocketException(int code, const char* message);
};

struct SocketIo {
    unsigned int socketIndex;
    const char* data;
    size_t size;
};

class SocketPool {
private:
    int* sockets;
    std::atomic_uint sentBytes;
    std::atomic_uint recvBytes;

public:
    static SocketPool* connect(unsigned int nSockets, char** hosts, int* ports);

    unsigned int nSockets;

    SocketPool(unsigned int nSockets, int* sockets);
    ~SocketPool();

    void setNotBlocking(bool enabled);
    void write(unsigned int socketIndex, const char* data, size_t size);
    void read(unsigned int socketIndex, char* data, size_t size);
    void writeMany(unsigned int n, SocketIo* ios);
    void readMany(unsigned int n, SocketIo* ios);
    void getStats(size_t* sentBytes, size_t* recvBytes);
};

class Socket {
private:
    int socket;

public:
    static Socket accept(int port);

    Socket(int socket);
    ~Socket();

    void setNotBlocking(bool enabled);
    void write(const char* data, size_t size);
    void read(char* data, size_t size);
};

#endif
