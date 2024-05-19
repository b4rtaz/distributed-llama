#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "../../utils.hpp"
#include "../../socket.hpp"
#include "../../transformer.hpp"
#include "../../tasks.hpp"
#include "../../tokenizer.hpp"
#include "../../app.hpp"

void generate(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, AppArgs* args, TransformerSpec* spec) {
    assert(args->prompt != NULL);

    // encode the (string) prompt into tokens sequence
    int numPromptTokens = 0;
    int* promptTokens = (int*)malloc((strlen(args->prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS

    // TODO: this is a hack for Grok1. We should have a more general way to handle this
    bool addBos = spec->archType != GROK1;

    tokenizer->encode(args->prompt, promptTokens, &numPromptTokens, addBos, false);
    if (numPromptTokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

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
        safePrintf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        if (args->benchmark)
            printf("\n");
        fflush(stdout);
        token = next;
    }

    free(promptTokens);

    if (!args->benchmark) printf("\n");
    double avgGenerationTime = totalGenerationTime / (double)pos;
    printf("Generated tokens:    %d\n", pos);
    printf("Avg tokens / second: %.2f\n", 1000.0 / avgGenerationTime);
    printf("Avg generation time: %.2f ms\n", avgGenerationTime);
    printf("Avg inference time:  %.2f ms\n", totalInferenceTime / (double)pos);
    printf("Avg transfer time:   %.2f ms\n", totalTransferTime / (double)pos);
}

void chat(Inference* inference, SocketPool* socketPool, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec) {
    char* cliSystemPrompt = NULL;
    char* cliUserPrompt = NULL;
    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char systemPrompt[512];
    char userPrompt[512];
    const size_t renderedPromptSize = 1152;
    char renderedPrompt[renderedPromptSize];
    int numPromptTokens = 0;
    int* promptTokens = (int*)malloc(1152 * sizeof(int));
    int userIdx;

    // start the main loop
    int8_t userTurn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    pos_t pos = 0;     // position in the sequence
    while (pos < args->steps) {
        // when it is the user's turn to contribute tokens to the dialog...
        if (userTurn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cliSystemPrompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    readStdin("💻 Enter system prompt (optional): ", systemPrompt, sizeof(systemPrompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(systemPrompt, cliSystemPrompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cliUserPrompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(userPrompt, cliUserPrompt);
            } else {
                // otherwise get user prompt from stdin
                readStdin("👱 User: ", userPrompt, sizeof(userPrompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && systemPrompt[0] != '\0') {
                char systemTemplate[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                snprintf(renderedPrompt, renderedPromptSize, systemTemplate, systemPrompt, userPrompt);
            } else {
                char userTemplate[] = "[INST] %s [/INST]";
                snprintf(renderedPrompt, renderedPromptSize, userTemplate, userPrompt);
            }
            // encode the rendered prompt into tokens
            tokenizer->encode(renderedPrompt, promptTokens, &numPromptTokens, true, false);
            userIdx = 0; // reset the user index
            userTurn = 0;
            printf("🤖 Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (userIdx < numPromptTokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = promptTokens[userIdx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS token ends the Assistant turn
        if (token == tokenizer->eosId) {
            userTurn = 1;
        }

        // forward the transformer to get logits for the next token
        float* logits = inference->infer(token, pos);
        next = sampler->sample(logits);
        pos++;

        if (userIdx >= numPromptTokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = tokenizer->decode(token, next);
            safePrintf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == tokenizer->eosId) { printf("\n"); }
    }
    printf("\n");
    free(promptTokens);
}

void worker(AppArgs* args) {
    if (args->port < 1024) {
        throw std::runtime_error("Invalid port number");
    }

    SocketServer server(args->port);
    Socket socket = server.accept();
    TransformerSpec spec;
    Transformer transformer = Transformer::loadSlice(&spec, &socket);
    TransformerArch arch = TransformerArchFactory::create(&spec);

    Worker worker = Worker(&arch, args->nThreads, &transformer, &socket);
    worker.work();
}

int main(int argc, char *argv[]) {
    initQuants();

    AppArgs args = AppArgs::parse(argc, argv, true);

    if (args.mode != NULL) {
        if (strcmp(args.mode, "inference") == 0) {
            args.benchmark = true;
            App::run(&args, generate);
            return EXIT_SUCCESS;
        } else if (strcmp(args.mode, "generate") == 0) {
            args.benchmark = false;
            App::run(&args, generate);
            return EXIT_SUCCESS;
        } else if (strcmp(args.mode, "chat") == 0) {
            App::run(&args, chat);
            return EXIT_SUCCESS;
        } else if (strcmp(args.mode, "worker") == 0) {
            worker(&args);
            return EXIT_SUCCESS;
        }
    }

    fprintf(stderr, "Invalid usage\n");
    return EXIT_FAILURE;
}
