#ifndef SOCKET_HPP
#define SOCKET_HPP

#include <atomic>
#include <cstddef>
#include <exception>
#include <vector>

#define ROOT_SOCKET_INDEX 0

void initSockets();
void cleanupSockets();
int acceptSocket(int serverSocket);
void setReuseAddr(int socket);
void writeSocket(int socket, const void* data, size_t size);
void readSocket(int socket, void* data, size_t size);
int createServerSocket(int port);
void closeServerSocket(int serverSocket);

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
    const void* data;
    size_t size;
    size_t extra;
};

class SocketPool {
private:
    int* sockets;
    std::atomic_uint sentBytes;
    std::atomic_uint recvBytes;
    size_t packetAlignment;
    char* packetAlignmentBuffer;

public:
    static SocketPool* serve(int port);
    static SocketPool* connect(unsigned int nSockets, char** hosts, int* ports, size_t packetAlignment);

    unsigned int nSockets;

    SocketPool(unsigned int nSockets, int* sockets, size_t packetAlignment);
    ~SocketPool();

    void setTurbo(bool enabled);
    void write(unsigned int socketIndex, const void* data, size_t size);
    void read(unsigned int socketIndex, void* data, size_t size);
    bool tryReadWithAlignment(unsigned int socketIndex, void* data, size_t size, unsigned long maxAttempts);
    void writeManyWithAlignment(unsigned int n, SocketIo* ios);
    void readManyWithAlignment(unsigned int n, SocketIo* ios);
    void getStats(size_t* sentBytes, size_t* recvBytes);

private:
    size_t calculateExtraBytesForAlignment(size_t size);
};

#endif
