#include <stdlib.h>
#include <string.h>
#include "funcs.hpp"
#include "shared-buffer.hpp"
#include "transformer.hpp"
#include "tokenizer.hpp"
#include "worker.hpp"

struct ProgramArgs {
    char* mode;

    // inference
    char* modelPath;
    char* tokenizerPath;
    char* prompt;
    FloatType floatType;
    int sliceCount;

    // worker
    int port;
};

int usage() {
    printf("Usage:\n");
    printf("main inference -m <model_path> -f <float_type> -t <tokenizer_path> -p <prompt> -s 1");
    return EXIT_FAILURE;
}

int inference(ProgramArgs* args) {
    if (args->modelPath == NULL || args->tokenizerPath == NULL || args->prompt == NULL) {
        return usage();
    }

    float temperature = 0.8f;
    float topp = 0.9f;
    int steps = 256;

    TransformerSpec* spec;
    Transformer* transformer;
    loadTransformer(&spec, &transformer, args->modelPath, args->floatType, args->sliceCount);

    generate(transformer, args->tokenizerPath, temperature, topp, steps, args->prompt);

    delete transformer;
    delete spec;
    return EXIT_SUCCESS;
}

int worker(ProgramArgs* args) {
    if (args->port < 1024) {
        return usage();
    }

    Worker::serve(args->port);
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    initQuants();

    ProgramArgs args;
    args.mode = NULL;
    args.modelPath = NULL;
    args.tokenizerPath = NULL;
    args.prompt = NULL;
    args.floatType = F32;
    args.sliceCount = 1;
    args.port = 9988;

    if (argc > 1) {
        args.mode = argv[1];
    }
    for (int i = 2; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "-m") == 0) {
            args.modelPath = argv[i + 1];
        } else if (strcmp(argv[i], "-t") == 0) {
            args.tokenizerPath = argv[i + 1];
        } else if (strcmp(argv[i], "-p") == 0) {
            args.prompt = argv[i + 1];
        } else if (strcmp(argv[i], "-f") == 0) {
            args.floatType = (FloatType)atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-s") == 0) {
            args.sliceCount = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-p") == 0) {
            args.port = atoi(argv[i + 1]);
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
