#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "utils.hpp"
#include "socket.hpp"
#include "transformer.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"
#include "mixtral-tasks.hpp"
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

TransformerArch getArch(TransformerSpec* spec) {
    if (spec->archType == LLAMA2) return buildLlama2Arch(spec);
    if (spec->archType == GROK1) return buildGrok1Arch(spec);
    if (spec->archType == MIXTRAL) return buildMixtralArch(spec);
    printf("Unsupported arch type: %d\n", spec->archType);
    exit(EXIT_FAILURE);
}

void generate(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, ProgramArgs* args, TransformerSpec* spec) {
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
    int pos = 0;     // position in the sequence

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
    
        printf("🔶 G %4ld ms I %4ld ms T %4ld ms S %6ld kB R %6ld kB ", generationTime, inferenceTime, transferTime, sentBytes / 1024, recvBytes / 1024);
        safePrintf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        printf("\n");
        fflush(stdout);
        token = next;
    }

    free(promptTokens);

    printf("Generated tokens:    %d\n", pos);
    printf("Avg generation time: %.2f ms\n", totalGenerationTime / (double)pos);
    printf("Avg inference time:  %.2f ms\n", totalInferenceTime / (double)pos);
    printf("Avg transfer time:   %.2f ms\n", totalTransferTime / (double)pos);
}

void chat(Inference* inference, SocketPool* socketPool, Tokenizer* tokenizer, Sampler* sampler, ProgramArgs* args, TransformerSpec* spec) {
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
    int pos = 0;     // position in the sequence
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

void simpleServer(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, ProgramArgs* args, TransformerSpec* spec) {
    int outputPos;
    int header[2];
    char output[8192]; // TODO: possible overflow if long context

    SocketServer server(args->port);
    while (true) {
        try {
            Socket client = server.accept();
            while (true) {
                outputPos = 0;
                printf("⏳\n");
                fflush(stdout);

                client.read((char*)&header, sizeof(int) * 2);
                int promptSize = header[0];
                int maxTokens = header[1];

                char prompt[promptSize + 1];
                client.read(prompt, promptSize);
                prompt[promptSize] = 0;
                printf("🚧 %s", prompt);
                fflush(stdout);

                int nPromptTokens;
                int promptTokens[promptSize + 3];
                tokenizer->encode(prompt, promptTokens, &nPromptTokens, true, false);

                int token = promptTokens[0];
                int maxPos = nPromptTokens + maxTokens;
                if (maxPos > spec->seqLen) maxPos = spec->seqLen;
                for (int pos = 0; pos < maxPos; pos++) {
                    float* logits = inference->infer(token, pos);

                    if (pos < nPromptTokens - 1) {
                        token = promptTokens[pos + 1];
                    } else {
                        int prevToken = token;
                        token = sampler->sample(logits);

                        if (token == tokenizer->bosId || token == tokenizer->eosId) {
                            printf("🚫");
                            fflush(stdout);
                            break;
                        }

                        char* piece = tokenizer->decode(prevToken, token);
                        if (isSafePiece(piece)) {
                            int pieceLen = strlen(piece);
                            memcpy(&output[outputPos], piece, pieceLen);
                            outputPos += pieceLen;
                        }
                        safePrintf(piece);
                        fflush(stdout);
                    }
                }

                output[outputPos] = 0;
                client.write((char*)&outputPos, sizeof(int));
                client.write(output, outputPos);
            }
        } catch (ReadSocketException& ex) {
            printf("Read socket error: %d %s\n", ex.code, ex.message);
        } catch (WriteSocketException& ex) {
            printf("Write socket error: %d %s\n", ex.code, ex.message);
        }
    }
}

int run(ProgramArgs* args, void (*program)(Inference* inference, SocketPool* socketPool, Tokenizer* tokenizer, Sampler* sampler, ProgramArgs* args, TransformerSpec* spec)) {
    if (args->modelPath == NULL) {
        return usage("Model is required");
    }
    if (args->tokenizerPath == NULL) {
        return usage("Tokenizer is required");
    }

    SocketPool* socketPool = SocketPool::connect(args->nWorkers, args->workerHosts, args->workerPorts);
    unsigned int nSlices = args->nWorkers + 1;
    unsigned long long rngSeed = (unsigned int)time(NULL);

    TransformerSpec spec = Transformer::loadSpecFromFile(args->modelPath, nSlices, args->weightsFloatType, args->bufferFloatType);
    TransformerArch arch = getArch(&spec);

    if (args->steps < 0) {
        args->steps = spec.seqLen;
    } else if (args->steps > spec.seqLen) {
        args->steps = spec.seqLen;
    }

    Tokenizer tokenizer(args->tokenizerPath, spec.vocabSize);
    Transformer transformer = Transformer::loadRootFromFile(args->modelPath, &spec, socketPool);
    Inference inference = Inference(&arch, args->nThreads, &transformer, socketPool);

    Sampler sampler(spec.vocabSize, args->temperature, args->topp, rngSeed);

    program(&inference, socketPool, &tokenizer, &sampler, args, &spec);

    delete socketPool;
    return EXIT_SUCCESS;
}

int worker(ProgramArgs* args) {
    if (args->port < 1024) {
        return usage("Invalid port");
    }

    SocketServer server(args->port);
    Socket socket = server.accept();
    TransformerSpec spec;
    Transformer transformer = Transformer::loadSlice(&spec, &socket);
    TransformerArch arch = getArch(&spec);

    Worker worker = Worker(&arch, args->nThreads, &transformer, &socket);
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
            return run(&args, generate);
        } else if (strcmp(args.mode, "chat") == 0) {
            return run(&args, chat);
        } else if (strcmp(args.mode, "simple-server") == 0) {
            return run(&args, simpleServer);
        } else if (strcmp(args.mode, "worker") == 0) {
            return worker(&args);
        }
    }
    return usage("Unknown mode");
}
