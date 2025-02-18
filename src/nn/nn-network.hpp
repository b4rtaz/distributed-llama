#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn-executor.hpp"

#define ROOT_SOCKET_INDEX 0

void initSockets();
void cleanupSockets();
int acceptSocket(int serverSocket);
void setReuseAddr(int socket);
void writeSocket(int socket, const void* data, NnSize size);
void readSocket(int socket, void* data, NnSize size);
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
    NnUint socketIndex;
    const void *data;
    NnSize size;
};

class NnNetwork {
private:
    int *sockets;
    NnSize *sentBytes;
    NnSize *recvBytes;

public:
    static std::unique_ptr<NnNetwork> serve(int port);
    static std::unique_ptr<NnNetwork> connect(NnUint nSockets, char **hosts, NnUint *ports);

    NnUint nSockets;

    NnNetwork(NnUint nSockets, int *sockets);
    ~NnNetwork();

    void setTurbo(bool enabled);
    void write(const NnUint socketIndex, const void *data, const NnSize size);
    void read(const NnUint socketIndex, void *data, const NnSize size);
    void writeAck(const NnUint socketIndex);
    void readAck(const NnUint socketIndex);
    bool tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts);
    void writeMany(NnUint n, NnSocketIo *ios);
    void writeAll(void *data, NnSize size);
    void readMany(NnUint n, NnSocketIo *ios);
    void getStats(NnSize *sentBytes, NnSize *recvBytes);
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
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

class NnRootConfigWriter {
private:
    NnNetwork *network;
public:
    NnRootConfigWriter(NnNetwork *network);
    void writeNet(NnUint socketIndex, NnNetConfig *config);
    void writeNode(NnUint socketIndex, NnNodeConfig *config);
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
    NnUint nNodes;
    NnByte *temp;
    NnSize tempSize;
public:
    NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes);
    ~NnRootWeightLoader();
    void writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRowMatmulSlices(const char *opName, NnUint opIndex, NnRowMatmulSlice *slice, NnByte *weight);
    NnSize loadColMatmulSlices(const char *opName, NnUint opIndex, NnColMatmulSlice *slice, NnByte *weight);
    void finish();
private:
    void allocate(NnSize size);};

class NnWorkerWeightReader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnByte *temp;
    NnUint tempSize;
public:
    NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network);
    ~NnWorkerWeightReader();
    void read();
private:
    void allocate(NnUint size);
};

#endif
