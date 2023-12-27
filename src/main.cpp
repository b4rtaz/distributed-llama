#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include "utils.hpp"
#include "socket.hpp"
#include "transformer.hpp"
#include "transformer-tasks.hpp"
#include "tokenizer.hpp"

struct ProgramArgs {
    char* mode;
    int nThread; 

    // inference
    char* modelPath;
    char* tokenizerPath;
    char* prompt;
    FloatType floatType;
    int nWorkers;
    char** workerHosts;
    int* workerPorts;

    // worker
    int port;
};

int usage() {
    printf("Invalid usage\n");
    // TODO
    return EXIT_FAILURE;
}

int inference(ProgramArgs* args) {
    if (args->modelPath == NULL || args->tokenizerPath == NULL || args->prompt == NULL) {
        return usage();
    }

    float temperature = 0.8f;
    float topp = 0.9f;
    int steps = 256;

    SocketPool socketPool = SocketPool::connect(args->nWorkers, args->workerHosts, args->workerPorts);
    unsigned int nSlices = args->nWorkers + 1;

    TransformerSpec spec = Transformer::loadSpecFromFile(args->modelPath, nSlices, args->floatType);
    Transformer transformer = Transformer::loadRootFromFile(args->modelPath, &spec, &socketPool);
    Inference inference = Inference(args->nThread, &transformer, &socketPool);

    socketPool.setTurboMode();

    generate(&spec, &inference, args->tokenizerPath, temperature, topp, steps, args->prompt);

    return EXIT_SUCCESS;
}

int worker(ProgramArgs* args) {
    if (args->port < 1024) {
        return usage();
    }

    Socket socket = Socket::accept(args->port);
    TransformerSpec spec;
    Transformer transformer = Transformer::loadSlice(&spec, &socket);

    socket.setTurboMode();

    Worker worker = Worker(args->nThread, &transformer, &socket);
    worker.work();

    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    initQuants();

    ProgramArgs args;
    args.mode = NULL;
    args.nThread = 4;
    args.modelPath = NULL;
    args.tokenizerPath = NULL;
    args.prompt = NULL;
    args.floatType = F32;
    args.nWorkers = 0;
    args.port = 9990;

    if (argc > 1) {
        args.mode = argv[1];
    }
    for (int i = 2; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "-nthread") == 0) {
            args.nThread = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-m") == 0) {
            args.modelPath = argv[i + 1];
        } else if (strcmp(argv[i], "-t") == 0) {
            args.tokenizerPath = argv[i + 1];
        } else if (strcmp(argv[i], "-prompt") == 0) {
            args.prompt = argv[i + 1];
        } else if (strcmp(argv[i], "-f") == 0) {
            args.floatType = (FloatType)atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-s") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new int[count];

            for (int s = 0; s < count; s++) {
                char* v = argv[i + 1 + s];
                char* sep = strstr(v, ":");
                if (sep == NULL) {
                    printf("Invalid address %s\n", v);
                    exit(EXIT_FAILURE);
                }
                int hostLen = sep - v;
                args.workerHosts[s] = new char[hostLen + 1];
                memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = atoi(sep + 1);
            }

            i += count;
        } else if (strcmp(argv[i], "-p") == 0) {
            args.port = atoi(argv[i + 1]);
        } else {
            printf("Unknown option %s\n", argv[i]);
        }
    }

    if (args.mode != NULL) {
        if (strcmp(args.mode, "inference") == 0) {
            return inference(&args);
        } else if (strcmp(args.mode, "worker") == 0) {
            return worker(&args);
        }
    }
    return usage();
}
