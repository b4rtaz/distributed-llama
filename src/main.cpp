#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include "utils.hpp"
#include "socket.hpp"
#include "transformer.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"
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

int usage(const char* reason) {
    printf("Invalid usage: %s\n", reason);
    return EXIT_FAILURE;
}

TransformerArch* getArch(TransformerArchType archType) {
    if (archType == LLAMA2) return &Llama2::arch;
    if (archType == GROK1) return &Grok1::arch;
    printf("Unsupported arch type: %d\n", archType);
    exit(EXIT_FAILURE);
}

int inferenceOrChat(ProgramArgs* args, bool isChat) {
    if (args->modelPath == NULL) {
        return usage("Model is required");
    }
    if (args->tokenizerPath == NULL) {
        return usage("Tokenizer is required");
    }
    if (!isChat && args->prompt == NULL) {
        return usage("Prompt is required");
    }

    SocketPool* socketPool = SocketPool::connect(args->nWorkers, args->workerHosts, args->workerPorts);
    unsigned int nSlices = args->nWorkers + 1;
    unsigned long long rngSeed = (unsigned int)time(NULL);

    TransformerSpec spec = Transformer::loadSpecFromFile(args->modelPath, nSlices, args->weightsFloatType, args->bufferFloatType);
    TransformerArch* arch = getArch(spec.archType);

    int steps = args->steps;
    if (steps < 0) {
        steps = spec.seqLen;
    } else if (steps > spec.seqLen) {
        steps = spec.seqLen;
    }

    Transformer transformer = Transformer::loadRootFromFile(args->modelPath, &spec, socketPool);
    Inference inference = Inference(arch, args->nThreads, &transformer, socketPool);

    bool bos = spec.archType == LLAMA2;
    bool eos = false;
    Tokenizer tokenizer(args->tokenizerPath, spec.vocabSize, bos, eos);
    Sampler sampler(spec.vocabSize, args->temperature, args->topp, rngSeed);

    socketPool->enableTurbo();

    if (isChat) {
        chat(&inference, &tokenizer, &sampler, NULL, NULL, steps);
    } else {
        generate(&spec, &inference, socketPool, &tokenizer, &sampler, steps, args->prompt);
    }

    delete socketPool;

    return EXIT_SUCCESS;
}

int worker(ProgramArgs* args) {
    if (args->port < 1024) {
        return usage("Invalid port");
    }

    Socket socket = Socket::accept(args->port);
    TransformerSpec spec;
    Transformer transformer = Transformer::loadSlice(&spec, &socket);
    TransformerArch* arch = getArch(spec.archType);

    socket.enableTurbo();

    Worker worker = Worker(arch, args->nThreads, &transformer, &socket);
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
    args.steps = -1;

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

            i += count - 1;
        } else if (strcmp(argv[i], "--port") == 0) {
            args.port = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--nthreads") == 0) {
            args.nThreads = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--steps") == 0) {
            args.steps = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--temperature") == 0) {
            args.temperature = atof(argv[i + 1]);
        } else if (strcmp(argv[i], "--topp") == 0) {
            args.topp = atof(argv[i + 1]);
        } else {
            printf("Unknown option %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }

    if (args.mode != NULL) {
        if (strcmp(args.mode, "inference") == 0) {
            return inferenceOrChat(&args, false);
        } else if (strcmp(args.mode, "chat") == 0) {
            return inferenceOrChat(&args, true);
        } else if (strcmp(args.mode, "worker") == 0) {
            return worker(&args);
        }
    }
    return usage("Unknown mode");
}
