#ifndef APP_HPP
#define APP_HPP

#include "quants.hpp"
#include "transformer.hpp"
#include "utils.hpp"
#include "utils.hpp"
#include "app.hpp"
#include "transformer.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"
#include "mixtral-tasks.hpp"
#include "tokenizer.hpp"

class AppArgs {
public:
    char* mode;
    int nThreads;
    bool useDiscForKvCache;

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
    pos_t steps;
    bool benchmark;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    unsigned int maxSeqLen;

    // worker
    int port;

    static AppArgs parse(int argc, char** argv, bool hasMode);
};

class TransformerArchFactory {
public:
    static TransformerArch create(TransformerSpec* spec);
};

class App {
public:
    static void run(AppArgs* args, void (*program)(Inference* inference, SocketPool* socketPool, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec));
};

#endif
