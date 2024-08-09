#ifndef SOCKET_HPP
#define SOCKET_HPP

// 主要定义了Socket通信的一些结构和函数,包括异常函数等等.


#include <atomic>
#include <cstddef>
// 指针操作库
#include <exception>
// 异常库
#include <vector>
// 动态数组库

void initSockets();
// 初始化Sockets
void cleanupSockets();
// 清除某个Sockets

class ReadSocketException : public std::exception {
public:
    int code;
    const char* message;
    ReadSocketException(int code, const char* message);
};
#if 0
定义ReadSocketException类,用于异常处理,其继承于在exception中的 std::exception类
code是异常编码
message是异常消息
ReadSocketException(int code, const char* message); --> 构造函数
#endif

class WriteSocketException : public std::exception {
public:
    int code;
    const char* message;
    WriteSocketException(int code, const char* message);
};
// 类似上面


//     unsigned int socketIndex; const void* data;
struct SocketIo {
    unsigned int socketIndex;
    const void* data;
    size_t size; // size_t是在cstddf头文件中引入的标准定义
};
// SocketIo结构体

class SocketPool {
private:
    int* sockets;
    std::atomic_uint sentBytes; // 用于统计发送的字节数
    std::atomic_uint recvBytes; // 用于统计接收的字节数

public:
    static SocketPool* connect(unsigned int nSockets, char** hosts, int* ports);
    // 静态成员函数使用实例化后的对象进行调用,语法为: SocketPool* pool = SocketPool::connect(nSockets, hosts, ports);
    // 如果Connect成功则返回SocketPool*指针
    unsigned int nSockets;

    SocketPool(unsigned int nSockets, int* sockets);
    ~SocketPool();

    void setTurbo(bool enabled); // 设置Turbo模式
    void write(unsigned int socketIndex, const void* data, size_t size); // 写
    void read(unsigned int socketIndex, void* data, size_t size); // 读
    void writeMany(unsigned int n, SocketIo* ios); 
    void readMany(unsigned int n, SocketIo* ios); // 同时向多个socket写入数据 | 从多个socket读取数据, ios是要写入 | 读取的对象数组
    void getStats(size_t* sentBytes, size_t* recvBytes); // 获取状态,得到上面的统计的收发字节数
};

class Socket {
private:
    int socket;

public:
    Socket(int socket); // 接收一个int类型的socket用于初始化变量
    ~Socket();

    void setTurbo(bool enabled); // 是否使用turbo模式
    void write(const void* data, size_t size); // 写入数据和大小,data是指向写入数据的指针
    void read(void* data, size_t size); // 读数据
    bool tryRead(void* data, size_t size, unsigned long maxAttempts); // 尝试读数据,及其最大读次数
    std::vector<char> readHttpRequest();
};

class SocketServer {
private:
    int socket;
public:
    SocketServer(int port);
    ~SocketServer();
    Socket accept(); // Server接收客户端的连接请求后调用,返回一个Socket对象,可以进行后续的通信
};

#endif
