#include <cstring>
#include <cstdlib>
#include <cstdint>
#include "funcs.hpp"
#include "shared-buffer.hpp"
#include "transformer.hpp"
#include "tokenizer.hpp"
#include "worker.hpp"

struct ProgramArgs {
    char* mode;
    int nThread; 

    // inference
    char* modelPath;
    char* tokenizerPath;
    char* prompt;
    FloatType floatType;
    int sliceCount;
    char** sliceHosts;
    int* slicePorts;

    // worker
    int port;
};

int usage() {
    printf("Usage:\n");
    printf("main inference -m <model_path> -f <float_type> -t <tokenizer_path> -prompt <prompt> -s 192.0.0.1:2000\n");
    return EXIT_FAILURE;
}

void loadConfig(ProgramArgs* args, TransformerConfig* config) {
    config->nThread = args->nThread;
    config->sliceCount = args->sliceCount;
    config->sliceHosts = args->sliceHosts;
    config->slicePorts = args->slicePorts;
}

int inference(ProgramArgs* args) {
    if (args->modelPath == NULL || args->tokenizerPath == NULL || args->prompt == NULL) {
        return usage();
    }

    float temperature = 0.8f;
    float topp = 0.9f;
    int steps = 256;

    TransformerSpec spec;
    loadTransformerSpec(&spec, args->modelPath, args->floatType, args->sliceCount);

    TransformerConfig config;
    loadConfig(args, &config);

    Transformer* transformer;
    loadTransformer(&transformer, &spec, &config, args->modelPath);

    generate(transformer, args->tokenizerPath, temperature, topp, steps, args->prompt);

    delete transformer;
    return EXIT_SUCCESS;
}

int worker(ProgramArgs* args) {
    if (args->port < 1024) {
        return usage();
    }

    TransformerConfig config;
    loadConfig(args, &config);

    Worker::serve(&config, args->port);
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
    args.sliceCount = 0;
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

            args.sliceCount = count;
            args.sliceHosts = new char*[count];
            args.slicePorts = new int[count];

            for (int s = 0; s < count; s++) {
                char* v = argv[i + 1 + s];
                if (strcmp(v, "local") == 0) {
                    args.sliceHosts[s] = NULL;
                    args.slicePorts[s] = -1;
                    continue;
                }

                char* sep = strstr(v, ":");
                if (sep == NULL) {
                    printf("Invalid address %s\n", v);
                    exit(EXIT_FAILURE);
                }
                int hostLen = sep - v;
                args.sliceHosts[s] = new char[hostLen + 1];
                memcpy(args.sliceHosts[s], v, hostLen);
                args.sliceHosts[s][hostLen] = '\0';
                args.slicePorts[s] = atoi(sep + 1);
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
            if (args.sliceCount == 0) {
                args.sliceCount = 1;
                args.sliceHosts = new char*[1];
                args.slicePorts = new int[1];
                args.sliceHosts[0] = NULL;
            }

            return inference(&args);
        } else if (strcmp(args.mode, "worker") == 0) {
            return worker(&args);
        }
    }
    return usage();
}
