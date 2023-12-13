#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>
#include "worker.hpp"

void Worker::listenClient(int clientSocket) {
    char buffer[1024];
    for (;;) {
        int bytes = recv(clientSocket, buffer, sizeof(buffer), 0);
        if (bytes < 0) {
            printf("Error reading from socket\n");
            exit(EXIT_FAILURE);
        }

        printf("Received %d bytes\n", bytes);
    }
}

void Worker::serve(int port) {
    const char* host = "127.0.0.1";
    int serverSocket;
    struct sockaddr_in serverAddr;
    struct sockaddr_in clientAddr;

    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        printf("Error creating socket\n");
        exit(EXIT_FAILURE);
    }

    memset(&serverAddr, '\0', sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = port;
    serverAddr.sin_addr.s_addr = inet_addr(host);

    int bindResult = bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr));
    if (bindResult < 0) {
        printf("Cannot bind %s:%d\n", host, port);
        exit(EXIT_FAILURE);
    }

    listen(serverSocket, 5);
    printf("Listening on %s:%d...\n", host, port);

    for(;;) {
        socklen_t clientAddrSize = sizeof(clientAddr);
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrSize);
        if (clientSocket < 0) {
            printf("Error accepting connection\n");
            exit(EXIT_FAILURE);
        }

        printf("Client connected\n");
        listenClient(clientSocket);
        printf("Client disconnected\n");
    }
}
