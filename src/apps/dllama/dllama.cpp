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
    if (args->prompt == NULL)
        throw std::runtime_error("Prompt is required");

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
            tokenizer->encode((char*)inputPrompt.c_str(), inputTokens, &nInputTokens, true, false);

            pos_t userPromptEndPos = (pos_t)std::min<unsigned int>(spec->seqLen, pos + nInputTokens - 1);
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
    if (args->port < 1024) {
        throw std::runtime_error("Invalid port number");
    }

    TransformerConfig config;
    config.useDiscForKvCache = args->useDiscForKvCache;

    SocketServer server(args->port);
    Socket socket = server.accept();
    TransformerSpec spec;
    Transformer transformer = Transformer::loadSlice(&spec, &config, &socket);
    TransformerArch arch = TransformerArchFactory::create(&spec);

    Worker worker = Worker(&arch, args->nThreads, &transformer, &socket);
    worker.work();
}

int main(int argc, char *argv[]) {
    initQuants();
    initSockets();

    AppArgs args = AppArgs::parse(argc, argv, true);
    bool success = false;

    if (args.mode != NULL) {
        if (strcmp(args.mode, "inference") == 0) {
            args.benchmark = true;
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

    cleanupSockets();

    if (success)
        return EXIT_SUCCESS;
    fprintf(stderr, "Invalid usage\n");
    return EXIT_FAILURE;
}
