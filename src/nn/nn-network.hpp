#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn-executor.hpp"

#define ROOT_SOCKET_INDEX 0

void initSockets();
void cleanupSockets();
int acceptSocket(int serverSocket);
void setReuseAddr(int socket);
void writeSocket(int socket, const void* data, size_t size);
void readSocket(int socket, void* data, size_t size);
int createServerSocket(int port);
void closeServerSocket(int serverSocket);

class NnReadNetworkException : public std::exception {
public:
    int code;
    const char *message;
    NnReadNetworkException(int code, const char *message);
};

class NnWriteNetworkException : public std::exception {
public:
    int code;
    const char *message;
    NnWriteNetworkException(int code, const char *message);
};

struct NnSocketIo {
    NnSize socketIndex;
    const void *data;
    size_t size;
};

class NnNetwork {
private:
    int *sockets;
    std::atomic_uint sentBytes;
    std::atomic_uint recvBytes;

public:
    static std::unique_ptr<NnNetwork> serve(int port);
    static std::unique_ptr<NnNetwork> connect(NnSize nSockets, char **hosts, NnSize *ports);

    NnSize nSockets;

    NnNetwork(NnSize nSockets, int *sockets);
    ~NnNetwork();

    void setTurbo(bool enabled);
    void write(NnSize socketIndex, const void *data, size_t size);
    void read(NnSize socketIndex, void *data, size_t size);
    void writeAck(NnSize socketIndex);
    void readAck(NnSize socketIndex);
    bool tryReadWithMaxAttempts(NnSize socketIndex, void *data, size_t size, unsigned long maxAttempts);
    void writeMany(NnSize n, NnSocketIo *ios);
    void writeAll(void *data, size_t size);
    void readMany(NnSize n, NnSocketIo *ios);
    void getStats(size_t *sentBytes, size_t *recvBytes);
    void resetStats();
};

class NnNetworkNodeSynchronizer : public NnNodeSynchronizer {
private:
    NnNetwork *network;
    NnNetExecution *execution;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
public:
    NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig);
    ~NnNetworkNodeSynchronizer() override {};
    void sync(NnSize segmentIndex, NnSize nThreads, NnSize threadIndex) override;
};

class NnRootConfigWriter {
private:
    NnNetwork *network;
public:
    NnRootConfigWriter(NnNetwork *network);
    void writeNet(NnSize socketIndex, NnNetConfig *config);
    void writeNode(NnSize socketIndex, NnNodeConfig *config);
    void writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs);
};

class NnWorkerConfigReader {
private:
    NnNetwork *network;
public:
    NnWorkerConfigReader(NnNetwork *network);
    NnNetConfig readNet();
    NnNodeConfig readNode();
};

class NnRootWeightLoader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnSize nNodes;
    NnByte *temp;
    NnSize tempSize;
public:
    NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnSize nNodes);
    ~NnRootWeightLoader();
    void writeWeight(NnSize nodeIndex, const char *opName, NnSize opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRoot(const char *opName, NnSize opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnSize opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRowMatmulSlices(const char *opName, NnSize opIndex, NnRowMatmulSlice *slice, NnByte *weight);
    NnSize loadColMatmulSlices(const char *opName, NnSize opIndex, NnColMatmulSlice *slice, NnByte *weight);
    void finish();
private:
    void allocate(NnSize size);};

class NnWorkerWeightReader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnByte *temp;
    NnSize tempSize;
public:
    NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network);
    ~NnWorkerWeightReader();
    void read();
private:
    void allocate(NnSize size);
};

#endif
