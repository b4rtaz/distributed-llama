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
    int nThreads; 

    // inference
    char* modelPath;
    char* tokenizerPath;
    char* prompt;
    FloatType weightsFloatType;
    FloatType bufferFloatType;
    int nWorkers;
    char** workerHosts;
    int* workerPorts;
    float temperature;
    float topp;
    int steps;

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

    SocketPool socketPool = SocketPool::connect(args->nWorkers, args->workerHosts, args->workerPorts);
    unsigned int nSlices = args->nWorkers + 1;

    TransformerSpec spec = Transformer::loadSpecFromFile(args->modelPath, nSlices, args->weightsFloatType, args->bufferFloatType);

    int steps = args->steps;
    if (steps > spec.seqLen) {
        steps = spec.seqLen;
    }

    Transformer transformer = Transformer::loadRootFromFile(args->modelPath, &spec, &socketPool);
    Inference inference = Inference(args->nThreads, &transformer, &socketPool);

    socketPool.enableTurbo();

    generate(&spec, &inference, &socketPool, args->tokenizerPath, args->temperature, args->topp, steps, args->prompt);

    return EXIT_SUCCESS;
}

int worker(ProgramArgs* args) {
    if (args->port < 1024) {
        return usage();
    }

    Socket socket = Socket::accept(args->port);
    TransformerSpec spec;
    Transformer transformer = Transformer::loadSlice(&spec, &socket);

    socket.enableTurbo();

    Worker worker = Worker(args->nThreads, &transformer, &socket);
    worker.work();

    return EXIT_SUCCESS;
}

FloatType parseFloatType(char* val) {
    if (strcmp(val, "f32") == 0) return F32;
    if (strcmp(val, "f16") == 0) return F16;
    if (strcmp(val, "q40") == 0) return Q40;
    if (strcmp(val, "q80") == 0) return Q80;
    printf("Invalid float type %s\n", val);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    initQuants();

    ProgramArgs args;
    args.mode = NULL;
    args.nThreads = 4;
    args.modelPath = NULL;
    args.tokenizerPath = NULL;
    args.prompt = NULL;
    args.weightsFloatType = F32;
    args.bufferFloatType = F32;
    args.nWorkers = 0;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 64;

    if (argc > 1) {
        args.mode = argv[1];
    }
    for (int i = 2; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--model") == 0) {
            args.modelPath = argv[i + 1];
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            args.tokenizerPath = argv[i + 1];
        } else if (strcmp(argv[i], "--prompt") == 0) {
            args.prompt = argv[i + 1];
        } else if (strcmp(argv[i], "--weights-float-type") == 0) {
            args.weightsFloatType = parseFloatType(argv[i + 1]);
        } else if (strcmp(argv[i], "--buffer-float-type") == 0) {
            args.bufferFloatType = parseFloatType(argv[i + 1]);
        } else if (strcmp(argv[i], "--workers") == 0) {
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
        } else if (strcmp(argv[i], "--port") == 0) {
            args.port = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--nthreads") == 0) {
            args.nThreads = atoi(argv[i + 1]);
        } else {
            printf("Unknown option %s\n", argv[i]);
            exit(EXIT_FAILURE);
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
