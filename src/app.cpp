#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <ctime>
#include <typeinfo>
#include "app.hpp"


FloatType parseFloatType(char* val) {
    if (strcmp(val, "f32") == 0) return F32; // --> 判断val和f32是否完全相同
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
    // 设置默认初始值
    int i = 1;
    if (hasMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    // argc是什么
    // 也就是在调用推理的时候,传入的 --model ... --tokenizer ... --prompt ... 实际上是一个数组,也就是这里的argv
    for (; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--model") == 0) { // 模型地址
            args.modelPath = argv[i + 1];
        } else if (strcmp(argv[i], "--tokenizer") == 0) { // tokenizer地址
            args.tokenizerPath = argv[i + 1];
        } else if (strcmp(argv[i], "--prompt") == 0) { // 提示词
            args.prompt = argv[i + 1];
        } else if (strcmp(argv[i], "--weights-float-type") == 0) { // 权重数据类型
            args.weightsFloatType = parseFloatType(argv[i + 1]);
        } else if (strcmp(argv[i], "--buffer-float-type") == 0) { // buffer数据类型
            args.bufferFloatType = parseFloatType(argv[i + 1]);
        } else if (strcmp(argv[i], "--workers") == 0) { // Workers Node
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++); // 得到Worker参数的终止位置
            int count = j - i - 1; // Worker个数

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new int[count];

            for (int s = 0; s < count; s++) {
                char* v = argv[i + 1 + s];
                char* sep = strstr(v, ":"); // x.x.x.x:y
                if (sep == NULL) {
                    printf("Invalid address %s\n", v);
                    exit(EXIT_FAILURE);
                }
                int hostLen = sep - v; // 一个IP地址的长度
                args.workerHosts[s] = new char[hostLen + 1];
                memcpy(args.workerHosts[s], v, hostLen); // 将从地址v开始的长度为hostLen的Bytes(也就是IP地址)拷贝到args.workerHosts[s]
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = atoi(sep + 1); // 将sep+1指向的地址转化为Int赋值给port
            }

            i += count - 1;
        } else if (strcmp(argv[i], "--port") == 0) { // 可能出现一个PC的多个端口
            args.port = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--nthreads") == 0) { // 线程数
            args.nThreads = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--steps") == 0) { // 最大推理步骤
            args.steps = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "--temperature") == 0) { // ?
            args.temperature = atof(argv[i + 1]);
        } else if (strcmp(argv[i], "--topp") == 0) { // ?
            args.topp = atof(argv[i + 1]);
        } else if (strcmp(argv[i], "--seed") == 0) { // 种子
            args.seed = atoll(argv[i + 1]);
        } else if (strcmp(argv[i], "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(argv[i + 1]); // ---- START ----
        } else if (strcmp(argv[i], "--memory-budget") == 0) {
            // version 1:把Memory Budget作为参数输入
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;
            for (int s = 0; s < count; s++) {
                char* v = argv[i + 1 + s];
                int k = std::atoi(v);
                args.memoryBudgetArray.push_back(k);
            }
            // for (int value : args.memoryBudgetArray){
            //     printf("test %d\n", value);
            //     printf("type %s\n", typeid(value).name());
            // }
            i += count - 1;
        }// ----- END -----
        else {
            printf("Unknown option %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
        // 这段 if...else...有点ugly...
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

    
    SocketPool* socketPool = SocketPool::connect(args->nWorkers, args->workerHosts, args->workerPorts); // 建立Socket连接
    unsigned int nSlices = args->nWorkers + 1; // 切片数量 = 总Node数量


    TransformerSpec spec = Transformer::loadSpecFromFile(args->modelPath, nSlices, args->weightsFloatType, args->bufferFloatType, args->memoryBudgetArray); // 从模型文件中读取spec参数
    // ---
    TransformerArch arch = TransformerArchFactory::create(&spec); // 读取arch,其中arch主要是对任务进行分配
    // 在这里只是分配任务,没有运行任务

    Tokenizer tokenizer(args->tokenizerPath, spec.vocabSize); // Tokenizer

    if (args->steps == 0 || args->steps > spec.seqLen) {
        args->steps = spec.seqLen; // seqLen是从loadSpecFromFile中读取的,最大输出的序列长度
    }

    Transformer transformer = Transformer::loadRootFromFile(args->modelPath, &spec, socketPool);
    //
    socketPool->setTurbo(true);

    
    Inference inference = Inference(&arch, args->nThreads, &transformer, socketPool);
    // taskLoop在这里决定
    Sampler sampler(spec.vocabSize, args->temperature, args->topp, args->seed);
    // 上面都是在定义参数
    program(&inference, socketPool, &tokenizer, &sampler, args, &spec);
    delete socketPool;
    
}
