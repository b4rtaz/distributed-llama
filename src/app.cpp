#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <ctime>
#include "app.hpp"

FloatType parseFloatType(char* val) {
    if (strcmp(val, "f32") == 0) return F32;
    if (strcmp(val, "f16") == 0) return F16;
    if (strcmp(val, "q40") == 0) return Q40;
    if (strcmp(val, "q80") == 0) return Q80;
    printf("Invalid float type %s\n", val);
    exit(EXIT_FAILURE);
}

ChatTemplateType parseChatTemplateType(char* val) {
    if (strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (strcmp(val, "zephyr") == 0) return TEMPLATE_ZEPHYR;
    if (strcmp(val, "chatml") == 0) return TEMPLATE_CHATML;
    throw std::runtime_error("Invalid chat template type");

}

AppArgs AppArgs::parse(int argc, char** argv, bool hasMode) {
    AppArgs args;
    args.mode = NULL;
    args.nThreads = 4;
    args.modelPath = NULL;
    args.tokenizerPath = NULL;
    args.prompt = NULL;
    args.weightsFloatType = FUNK;
    args.bufferFloatType = F32;
    args.nWorkers = 0;
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(NULL);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.useDiscForKvCache = false;

    int i = 1;
    if (hasMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    for (; i + 1 < argc; i += 2) {
        char* name = argv[i];
        char* value = argv[i + 1];
        if (strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (strcmp(name, "--weights-float-type") == 0) {
            args.weightsFloatType = parseFloatType(value);
        } else if (strcmp(name, "--buffer-float-type") == 0) {
            args.bufferFloatType = parseFloatType(value);
        } else if (strcmp(name, "--workers") == 0) {
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
        } else if (strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (strcmp(name, "--kv-cache-storage") == 0) {
            args.useDiscForKvCache = strcmp(value, "disc") == 0;
        } else {
            printf("Unknown option %s\n", name);
            exit(EXIT_FAILURE);
        }
    }
    return args;
}

TransformerArch TransformerArchFactory::create(TransformerSpec* spec) {
    if (spec->archType == LLAMA) return buildLlamaArch(spec);
    if (spec->archType == GROK1) return buildGrok1Arch(spec);
    if (spec->archType == MIXTRAL) return buildMixtralArch(spec);
    printf("Unsupported arch type: %d\n", spec->archType);
    exit(EXIT_FAILURE);
}

void App::run(AppArgs* args, void (*program)(Inference* inference, SocketPool* socketPool, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec)) {
    if (args->modelPath == NULL) {
        throw std::runtime_error("Model is required");
    }
    if (args->tokenizerPath == NULL) {
        throw std::runtime_error("Tokenizer is required");
    }

    SocketPool* socketPool = SocketPool::connect(args->nWorkers, args->workerHosts, args->workerPorts);
    unsigned int nSlices = args->nWorkers + 1;

    TransformerSpec spec = Transformer::loadSpecFromFile(args->modelPath, nSlices, args->maxSeqLen, args->weightsFloatType, args->bufferFloatType);
    TransformerArch arch = TransformerArchFactory::create(&spec);
    Tokenizer tokenizer(args->tokenizerPath, spec.vocabSize);

    if (args->steps == 0 || args->steps > spec.seqLen) {
        args->steps = spec.seqLen;
    }

    TransformerConfig config;
    config.useDiscForKvCache = args->useDiscForKvCache;

    Transformer transformer = Transformer::loadRootFromFile(args->modelPath, &spec, &config, socketPool);
    socketPool->setTurbo(true);

    Inference inference = Inference(&arch, args->nThreads, &transformer, socketPool);

    Sampler sampler(spec.vocabSize, args->temperature, args->topp, args->seed);

    program(&inference, socketPool, &tokenizer, &sampler, args, &spec);

    delete socketPool;
}
