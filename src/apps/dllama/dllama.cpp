#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <string>

#include "../../utils.hpp"
#include "../../socket.hpp"
#include "../../transformer.hpp"
#include "../../tasks.hpp"
#include "../../tokenizer.hpp"
#include "../../app.hpp"

void generate(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, AppArgs* args, TransformerSpec* spec) {
    // 这里的args就是我们在命令行传入的参数
    /*
    FOR ROOT Node:
    ./dllama
    inference -- 推理模式
    --model /Users/fox/Desktop/暑期项目/Github/Distributed_Llama/models/llama3_8b_q40/dllama_model_llama3_8b_q40.m -- 模型权重文件
    --tokenizer /Users/fox/Desktop/暑期项目/Github/Distributed_Llama/models/llama3_8b_q40/dllama_tokenizer_llama3_8b_q40.t -- Tokenizer文件
    --buffer-float-type q80 -- 缓存数据类型 q80 | q40 | F32 | F16 ...,q80应该是对RAM占用最小的
    --prompt "You are a person" -- 提示词
    --steps 16 --nthreads 4 -- 最大步数,可以理解为最大Token数
    --workers 10.3.10.139:9998 -- Worker的IP地址和端口
    */
    if (args->prompt == NULL) // 如果没有提示词
        throw std::runtime_error("Prompt is required");

    // 对提示词进行编码
    int numPromptTokens = 0;
    int* promptTokens = new int[strlen(args->prompt) + 3]; // +3 for '\0', ?BOS, ?EOS
    // 对GROK1的特殊处理
    bool addBos = spec->archType != GROK1;

    tokenizer->encode(args->prompt, promptTokens, &numPromptTokens, addBos, false);
    if (numPromptTokens < 1)
        throw std::runtime_error("Expected at least 1 prompt token");
    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = promptTokens[0]; // kick off with the first token in the prompt
    pos_t pos = 0;     // position in the sequence

    unsigned long inferenceTime; // 记录推理时间
    unsigned long transferTime; // 记录Transfer时间
    size_t sentBytes; // 记录发送字节数
    size_t recvBytes; // 记录收到字节数

    // 记录时间
    unsigned long totalGenerationTime = 0;
    unsigned long totalInferenceTime = 0;
    unsigned long totalTransferTime = 0;
    while (pos < args->steps) { // 在达到最大输出提示词之前进行循环
        unsigned long startTime = timeMs();
        float* logits = inference->infer(token, pos);

        inference->getStats(&inferenceTime, &transferTime); // 获取执行时间
        socketPool->getStats(&sentBytes, &recvBytes); // 

        // 如果我们仍然在input内:强制下一个输入token就是input中的下一个提示词作为输入
        // 如果我们已经超过了input:则根据上面预测的logits获取下一个token作为输入
        if (pos < numPromptTokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = promptTokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler->sample(logits);
        }
        pos++;

        unsigned long generationTime = timeMs() - startTime;
        // 记录一个step的时间

        totalGenerationTime += generationTime;
        totalInferenceTime += inferenceTime;
        totalTransferTime += transferTime;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == tokenizer->bosId) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        char* piece = tokenizer->decode(token, next); // 对预测结果进行解码

        if (args->benchmark) // 是否启用输出
            printf("🔶 G %4ld ms I %4ld ms T %4ld ms S %6ld kB R %6ld kB ", generationTime, inferenceTime, transferTime, sentBytes / 1024, recvBytes / 1024);
        safePrintf(piece);
        if (args->benchmark)
            printf("\n");
        fflush(stdout); // 刷新输出缓存区
        token = next; // 更新token变量,下一次推理的输入变量就是更新后的token
    }

    delete[] promptTokens;

    if (!args->benchmark) printf("\n");
    double avgGenerationTime = totalGenerationTime / (double)pos;
    printf("Generated tokens:    %d\n", pos);
    printf("Avg tokens / second: %.2f\n", 1000.0 / avgGenerationTime);
    printf("Avg generation time: %.2f ms\n", avgGenerationTime);
    printf("Avg inference time:  %.2f ms\n", totalInferenceTime / (double)pos);
    printf("Avg transfer time:   %.2f ms\n", totalTransferTime / (double)pos);
}

size_t readStdin(const char* guide, char* buffer, size_t bufsize) {
    fflush(stdin);
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
            len--;
        }
        return len;
    }
    return 0;
}

class Chat {
private:
    Inference* inference;
    Tokenizer* tokenizer;
    Sampler* sampler;
    AppArgs* args;
    TransformerSpec* spec;
    ChatTemplate* chatTemplate;
    EosDetector* eosDetector;

public:
    Chat(Inference* inference, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec, EosDetector* eosDetector, ChatTemplate* chatTemplate) {
        this->inference = inference;
        this->tokenizer = tokenizer;
        this->sampler = sampler;
        this->args = args;
        this->spec = spec;
        this->eosDetector = eosDetector;
        this->chatTemplate = chatTemplate;
    }

    void chat() {
        char inputBuffer[2048];

        size_t sysPromptLength = readStdin("💻 System prompt (optional): ", inputBuffer, sizeof(inputBuffer));
        std::vector<ChatItem> deltaItems;
        if (sysPromptLength > 0) {
            deltaItems.push_back(ChatItem{"system", inputBuffer});
        }

        pos_t pos = 0;
        int token;
        do {
            size_t userPromptLength;
            do {
                userPromptLength = readStdin("\n👱 User\n> ", inputBuffer, sizeof(inputBuffer));
            } while (userPromptLength == 0);

            deltaItems.push_back(ChatItem{"user", inputBuffer});

            size_t nChatItems = deltaItems.size();
            ChatItem chatItems[nChatItems];
            for (size_t j = 0; j < nChatItems; j++) {
                chatItems[j].role = deltaItems[j].role;
                chatItems[j].message = deltaItems[j].message;
            }
            std::string inputPrompt = chatTemplate->generate(deltaItems.size(), chatItems, true);

            int* inputTokens = new int[inputPrompt.size() + 3];
            int nInputTokens;
            tokenizer->encode((char*)inputPrompt.c_str(), inputTokens, &nInputTokens, true, false);

            pos_t userPromptEndPos = (pos_t)std::min(spec->seqLen, pos + nInputTokens - 1);
            for (pos_t i = 0; pos < userPromptEndPos; pos++, i++) {
                inference->infer(inputTokens[i], pos);
                token = inputTokens[i + 1];
            }

            printf("\n🤖 Assistant\n");

            for (; pos < spec->seqLen; pos++) {
                int prevToken = token;
                float* logits = inference->infer(token, pos);
                token = sampler->sample(logits);
                char* piece = tokenizer->decode(prevToken, token);
                bool isSafe = isSafePiece(piece);
                EosDetectorType eosType = eosDetector->append(token, isSafe ? piece : "");
                if (eosType == NOT_EOS || eosType == EOS) {
                    char* delta = eosDetector->getDelta();
                    if (delta != NULL) {
                        printf("%s", delta);
                        fflush(stdout);
                    }
                    eosDetector->clear();
                }
                if (eosType == EOS) break;
            }

            inputPrompt.clear();
        } while (pos < spec->seqLen);

        printf("(end of context)\n");
    }
};

void chat(Inference* inference, SocketPool* socketPool, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec) {
    TokenizerChatStops stops(tokenizer);
    ChatTemplate chatTemplate(args->chatTemplateType, tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(tokenizer->chatEosId, stops.nStops, stops.stops, stops.maxStopLength, stops.maxStopLength);

    Chat chat(inference, tokenizer, sampler, args, spec, &eosDetector, &chatTemplate);
    chat.chat();
}

void worker(AppArgs* args) {
    if (args->port < 1024) { // 端口号限制
        throw std::runtime_error("Invalid port number");
    }
    // for workers:
    SocketServer server(args->port); // 启动Server
    Socket socket = server.accept(); // 接收数据
    TransformerSpec spec; // 实例化一个spec
    Transformer transformer = Transformer::loadSlice(&spec, &socket, args->memoryBudgetArray); // 从接收到的数据中加载transformer
    TransformerArch arch = TransformerArchFactory::create(&spec); // 创建工作流

    Worker worker = Worker(&arch, args->nThreads, &transformer, &socket);
    // HERE !!
    worker.work(); // 开始工作
}

int main(int argc, char *argv[]) {
    initQuants(); // 初始化浮点数类型
    initSockets(); // 初始化Sockets

    AppArgs args = AppArgs::parse(argc, argv, true);
    // 从命令行解析参数
    bool success = false;

    if (args.mode != NULL) {
        if (strcmp(args.mode, "inference") == 0) {
            args.benchmark = true; // 设置benchmark
            App::run(&args, generate);
            success = true;
        } else if (strcmp(args.mode, "generate") == 0) {
            args.benchmark = false;
            App::run(&args, generate);
            success = true;
        } else if (strcmp(args.mode, "chat") == 0) {
            App::run(&args, chat);
            success = true;
        } else if (strcmp(args.mode, "worker") == 0) {
            worker(&args);
            success = true;
        }
    }

    cleanupSockets(); //关闭Socket

    if (success)
        return EXIT_SUCCESS;
    fprintf(stderr, "Invalid usage\n");
    return EXIT_FAILURE;
}
