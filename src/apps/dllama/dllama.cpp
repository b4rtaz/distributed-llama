#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>

#include "../../utils.hpp"
#include "../../socket.hpp"
#include "../../transformer.hpp"
#include "../../tasks.hpp"
#include "../../tokenizer.hpp"
#include "../../app.hpp"

void generate(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, AppArgs* args, TransformerSpec* spec) {
    if (args->prompt == NULL)
        throw BadArgumentException("Prompt is required");

    // encode the (string) prompt into tokens sequence
    int numPromptTokens = 0;
    int* promptTokens = new int[strlen(args->prompt) + 3]; // +3 for '\0', ?BOS, ?EOS

    // TODO: this is a hack for Grok1. We should have a more general way to handle this
    bool addBos = spec->archType != GROK1;

    tokenizer->encode(args->prompt, promptTokens, &numPromptTokens, addBos, false);
    if (numPromptTokens < 1)
        throw std::runtime_error("Expected at least 1 prompt token");

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = promptTokens[0]; // kick off with the first token in the prompt
    pos_t pos = 0;     // position in the sequence

    unsigned long inferenceTime;
    unsigned long transferTime;
    size_t sentBytes;
    size_t recvBytes;
    unsigned long totalGenerationTime = 0;
    unsigned long totalInferenceTime = 0;
    unsigned long totalTransferTime = 0;
    while (pos < args->steps) {
        unsigned long startTime = timeMs();
        float* logits = inference->infer(token, pos);

        inference->getStats(&inferenceTime, &transferTime);
        socketPool->getStats(&sentBytes, &recvBytes);

        // advance the state machine
        if (pos < numPromptTokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = promptTokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler->sample(logits);
        }
        pos++;

        unsigned long generationTime = timeMs() - startTime;

        totalGenerationTime += generationTime;
        totalInferenceTime += inferenceTime;
        totalTransferTime += transferTime;

        // data-dependent terminating condition: the BOS token delimits sequences
        if (next == tokenizer->bosId) {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        char* piece = tokenizer->decode(token, next);

        if (args->benchmark)
            printf("🔶 G %4ld ms I %4ld ms T %4ld ms S %6ld kB R %6ld kB ", generationTime, inferenceTime, transferTime, sentBytes / 1024, recvBytes / 1024);
        safePrintf(piece);
        if (args->benchmark)
            printf("\n");
        fflush(stdout);
        token = next;
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
            bool addBos = pos == 0;
            tokenizer->encode((char*)inputPrompt.c_str(), inputTokens, &nInputTokens, addBos, false);

            pos_t userPromptEndPos = (pos_t)std::min<unsigned int>(spec->seqLen, pos + nInputTokens - 1);
            for (pos_t i = 0; pos < userPromptEndPos; pos++, i++) {
                inference->infer(inputTokens[i], pos);
                token = inputTokens[i + 1];
            }

            printf("\n🤖 Assistant\n");

            for (; pos < spec->seqLen; ) {
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
                pos++;
                if (eosType == EOS) break;
            }

            deltaItems.clear();
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
    if (args->port < 1024) {
        throw std::runtime_error("Invalid port number");
    }

    TransformerConfig config;

    SocketPool* socketPool = SocketPool::serve(args->port);
    TransformerSpec spec;
    Transformer transformer = Transformer::loadSlice(&spec, &config, socketPool);
    TransformerArch arch = TransformerArchFactory::create(&spec);

    Worker worker = Worker(&arch, args->nThreads, &transformer, socketPool);
    worker.work();

    delete socketPool;
}

#ifdef _WIN32
    #define EXECUTABLE_NAME "dllama.exe"
#else
    #define EXECUTABLE_NAME "dllama"
#endif

bool isValidMode(const char *mode) {
    if (mode == NULL) {
        return false;
    } else {
        return (strcmp(mode, "generate") == 0) ||
               (strcmp(mode, "inference") == 0) ||
               (strcmp(mode, "chat") == 0) ||
               (strcmp(mode, "worker") == 0);
    }
}

std::unordered_map<std::string, std::string> examples = {
{"generate", ""},
{"inference", R"(  sudo nice -n -20 ./dllama inference \
    --prompt "Super briefly describe the 80s - ten words."\
    --steps 32 --seed 12345 \
    --model dllama_model_llama3_2_3b_instruct_q40.m \
    --tokenizer dllama_tokenizer_llama3_2_3b_instruct_q40.t \
    --buffer-float-type q80 --nthreads 4 --max-seq-len 8192 \
    --workers 10.0.0.2:9998 10.0.0.3:9998 10.0.0.4:9998
)"},
{"chat", R"(  sudo nice -n -20 ./dllama chat \
    --model dllama_model_llama3_2_3b_instruct_q40.m \
    --tokenizer dllama_tokenizer_llama3_2_3b_instruct_q40.t \
    --buffer-float-type q80 --nthreads 4 --max-seq-len 8192 \
    --workers 10.0.0.2:9998 10.0.0.3:9998 10.0.0.4:9998
)"},
{"worker", R"(  sudo nice -n -20 ./dllama worker --port 9998 --nthreads 4
)"}};

std::string inference_and_generate_usage_string =
    R"( {inference|generate} {--model <path>} {--tokenizer <path>}
        {--prompt <p>}
        [--steps <s>]
        [--buffer-float-type {f32|f16|q40|q80}]
        [--weights-float-type {f32|f16|q40|q80}]
        [--max-seq-len <max>]
        [--nthreads <n>]
        [--workers <ip:port> ...]
        [--packet-alignment <pa>]
        [--temperature <temp>]
        [--topp <t>]
        [--seed <s>]
)";

std::unordered_map<std::string, std::string> usageText = {
{"generate", inference_and_generate_usage_string},
{"inference", inference_and_generate_usage_string},
{"chat", R"( chat {--model <path>} {--tokenizer <path>}
        [--buffer-float-type {f32|f16|q40|q80}]
        [--weights-float-type {f32|f16|q40|q80}]
        [--max-seq-len <max>]
        [--nthreads <n>]
        [--workers <ip:port> ...]
        [--packet-alignment <pa>]
        [--temperature <temp>]
        [--topp <t>]
        [--seed <s>]
        [--chat-template {llama2|llama3|zephyr|chatml}]
)"},
{"worker", R"( worker [--nthreads <n>] [--port <p>]
)"}};

#define MULTIPLE_USAGES_PREFIX "       "
#define SOLO_USAGE_PREFIX "Usage: "

void usage(const char *mode, bool solo=true) {
    if (!isValidMode(mode)) {
        fprintf(stderr, "Usage: %s {inference | generate | chat | worker} {ARGS}\n", EXECUTABLE_NAME);
        usage("inference", false);
        usage("chat", false);
        usage("worker", false);
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "%s", examples["worker"].c_str());
        fprintf(stderr, "%s", examples["chat"].c_str());
        fprintf(stderr, "%s", examples["inference"].c_str());
    } else {
        fprintf(stderr, "%s%s%s",
                solo ? SOLO_USAGE_PREFIX : MULTIPLE_USAGES_PREFIX,
                EXECUTABLE_NAME,
                usageText[mode].c_str());
        if (solo && (!examples[mode].empty())) {
            fprintf(stderr, "Example:\n");
            fprintf(stderr, "%s", examples[mode].c_str());
        }
    }
    fflush(stderr);
}

void usage() {
    usage(NULL);
}

int main(int argc, char *argv[]) {
    initQuants();
    initSockets();

    bool success = false;

    try {
        AppArgs args = AppArgs::parse(argc, argv, true);
        if (args.help) {
            if ((args.mode == NULL) ||
                (strcmp(args.mode, "--usage") == 0) ||
                (strcmp(args.mode, "--help") == 0) ||
                (strcmp(args.mode, "-h") == 0)) {
                usage();
            } else if (isValidMode(args.mode)) {
                usage(args.mode);
            } else {
                usage();
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        if (args.mode != NULL) {
            if (strcmp(args.mode, "inference") == 0) {
                if (args.prompt == NULL) {
                    throw BadArgumentException("Prompt is required");
                }
                args.benchmark = true;
                App::run(&args, generate);
                success = true;
            } else if (strcmp(args.mode, "generate") == 0) {
                if (args.prompt == NULL) {
                    throw BadArgumentException("Prompt is required");
                }
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
    } catch (const BadArgumentException& e) {
        fprintf(stderr, "%s\n\n", e.what());
        if ((argc > 1) && isValidMode(argv[1])) {
            usage(argv[1]);
        } else {
            usage();
        }
        cleanupSockets();
        return EXIT_FAILURE;
    }

    cleanupSockets();

    if (success)
        return EXIT_SUCCESS;
    fprintf(stderr, "Invalid usage\n\n");
    usage();
    return EXIT_FAILURE;
}
