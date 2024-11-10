#include "../../socket.hpp"
#include <chrono>
#include <cstring>
#include <cstdio>
using namespace std::chrono;

unsigned int packageSizes[] = { 128, 256, 512, 768, 1024, 1280, 1518, 2048, 4096, 8192, 16384, 32768, 65536 };
unsigned int packageSizesCount = sizeof(packageSizes) / sizeof(unsigned int);
unsigned int maPackageSize = packageSizes[packageSizesCount - 1];
unsigned int nAttempts = 5000;
int port = 7721;

void server() {
    printf("nAttempts: %d\n", nAttempts);
    SocketServer server(port);
    Socket socket = server.accept();
    char buffer[maPackageSize];

    for (long i = 0; i < packageSizesCount; i++) {
        unsigned int currentPackageSize = packageSizes[i];

        long long totalReadTime = 0;
        long long totalWriteTime = 0;
        long long totalTime = 0; // [us]
        for (long a = 0; a < nAttempts; a++) {
            auto t0 = high_resolution_clock::now();
            socket.read(buffer, currentPackageSize);
            auto t1 = high_resolution_clock::now();
            socket.write(buffer, currentPackageSize);
            auto t2 = high_resolution_clock::now();

            totalReadTime += duration_cast<microseconds>(t1 - t0).count();
            totalWriteTime += duration_cast<microseconds>(t2 - t1).count();
            totalTime += duration_cast<microseconds>(t2 - t0).count();
        }

        double nPingPongs = (1.0 / (totalTime / 1000000.0)) * (double)nAttempts;
        printf("[%6d bytes] write: %5lld us, read: %5lld us, total: %5lld us, nPingPongs: %.2f\n",
            currentPackageSize, totalWriteTime, totalReadTime, totalTime, nPingPongs);
    }
}

void client(char* host) {
    char** hosts = new char*[1];
    hosts[0] = host;
    int* ports = new int[1];
    ports[0] = port;

    SocketPool* pool = SocketPool::connect(1, hosts, ports);
    char buffer[maPackageSize];
    pool->setTurbo(true);

    for (long i = 0; i < packageSizesCount; i++) {
        unsigned int currentPackageSize = packageSizes[i];

        long long totalReadTime = 0;
        long long totalWriteTime = 0;
        long long totalTime = 0; // [us]
        for (long a = 0; a < nAttempts; a++) {
            auto t0 = high_resolution_clock::now();
            pool->write(0, buffer, currentPackageSize);
            auto t1 = high_resolution_clock::now();
            pool->read(0, buffer, currentPackageSize);
            auto t2 = high_resolution_clock::now();

            totalWriteTime += duration_cast<microseconds>(t1 - t0).count();
            totalReadTime += duration_cast<microseconds>(t2 - t1).count();
            totalTime += duration_cast<microseconds>(t2 - t0).count();
        }

        printf("[%6d bytes] write: %5lld us, read: %5lld us, total: %5lld us\n",
            currentPackageSize, totalWriteTime, totalReadTime, totalTime);
    }

    delete pool;
    delete[] hosts;
    delete[] ports;
}

int main(int argc, char *argv[]) {
    initSockets();
    if (argc > 1 && strcmp(argv[1], "server") == 0) {
        server();
    } else if (argc > 2 && strcmp(argv[1], "client") == 0) {
        client(argv[2]);
    } else {
        printf("Invalid arguments\n");
    }
}
