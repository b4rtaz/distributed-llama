#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"
#include <stdexcept>

static void inference(AppInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0)
        throw std::runtime_error("Number of steps is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnSize pos = 0;
    int token;
    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, true, false);

    if (nInputTokens > context->args->steps)
        throw std::runtime_error("The number of input tokens is greater than the number of steps");

    Timer evalTimer;
    size_t sentBytes = 0;
    size_t recvBytes = 0;
    printf("%s\n", context->args->prompt);
    for (;;) {
        Timer batchTimer;
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0)
            break;
        NnSize batchSize = remainingTokens < context->args->nBatches
            ? remainingTokens
            : context->args->nBatches;

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnSize i = 0; i < batchSize; i++)
            context->inference->setToken(i, inputTokens[pos + i]);

        context->inference->forward();

        pos += batchSize;
        token = inputTokens[pos + 1];

        if (context->network != nullptr)
            context->network->getStats(&sentBytes, &recvBytes);
        printf("🔷️ E%5u ms S%6zu kB R%6zu kB (%d tokens)\n",
            batchTimer.elapsed(),
            sentBytes / 1024,
            recvBytes / 1024,
            batchSize);
    }
    NnSize evalTime = evalTimer.elapsed();

    fflush(stdout);

    context->inference->setBatchSize(1);

    Timer predTimer;
    for (; pos <= context->args->steps; pos++) {
        Timer tokenTimer;
        unsigned int prevToken = token;
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        context->inference->forward();

        token = context->sampler->sample(context->inference->logitsPipe);

        char* piece = context->tokenizer->decode(prevToken, token);

        if (isSafePiece(piece)) {
            if (context->network != nullptr)
                context->network->getStats(&sentBytes, &recvBytes);
        
            printf("🔶 P%5u ms S%6zu kB R%6zu kB %s\n",
                tokenTimer.elapsed(),
                sentBytes / 1024,
                recvBytes / 1024,
                piece);
            fflush(stdout);
        }
    }
    NnSize predTime = predTimer.elapsed();

    NnSize nEvalTokens = nInputTokens - 1;
    NnSize nPredTokens = pos - nEvalTokens;
    printf("\n");
    printf("Evaluation\n");
    printf("   nBatches: %d\n", context->args->nBatches);
    printf("    nTokens: %d\n", nEvalTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        nEvalTokens / (evalTime / 1000.0),
        evalTime / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("    nTokens: %d\n", nPredTokens);
    printf("   tokens/s: %3.2f (%3.2f ms/tok)\n",
        nPredTokens / (predTime / 1000.0),
        predTime / ((float) nPredTokens));
}

static size_t readStdin(const char *guide, char *buffer, size_t size) {
    std::fflush(stdin);
    std::printf("%s", guide);
    if (std::fgets(buffer, size, stdin) != NULL) {
        size_t length = std::strlen(buffer);
        if (length > 0 && buffer[length - 1] == '\n') {
            buffer[length - 1] = '\0';
            length--;
        }
        return length;
    }
    return 0;
}

static void chat(AppInferenceContext *context) {
    const NnSize seqLen = context->header->seqLen;
    char prompt[2048];

    TokenizerChatStops stops(context->tokenizer);
    ChatTemplate chatTemplate(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(context->tokenizer->chatEosId, stops.nStops, stops.stops, stops.maxStopLength, stops.maxStopLength);

    const size_t sysPromptLength = readStdin("💻 System prompt (optional): ", prompt, sizeof(prompt));
    std::vector<ChatItem> deltaItems;
    if (sysPromptLength > 0)
        deltaItems.push_back(ChatItem{"system", prompt});

    NnSize pos = 0;
    size_t userPromptLength;
    int token;
    int nInputTokens;
    do {
        do {
            userPromptLength = readStdin("\n👱 User\n> ", prompt, sizeof(prompt));
        } while (userPromptLength == 0);

        deltaItems.push_back(ChatItem{"user", prompt});

        std::string inputPrompt = chatTemplate.generate(deltaItems.size(), deltaItems.data(), true);
        std::unique_ptr<int[]> inputTokensPtr(new int[inputPrompt.size() + 3]);
        int *inputTokens = inputTokensPtr.get();

        bool addBos = pos == 0;
        context->tokenizer->encode((char*)inputPrompt.c_str(), inputTokens, &nInputTokens, addBos, false);

        NnSize userPromptEndPos = (NnSize)std::min<unsigned int>(seqLen, pos + nInputTokens - 1);
        for (;;) {
            int remainingTokens = userPromptEndPos - pos;
            if (remainingTokens <= 0)
                break;
            NnSize batchSize = remainingTokens < context->args->nBatches
                ? remainingTokens
                : context->args->nBatches;

            context->inference->setBatchSize(batchSize);
            context->inference->setPosition(pos);
            for (NnSize i = 0; i < batchSize; i++)
                context->inference->setToken(i, inputTokens[i]);

            context->inference->forward();

            pos += batchSize;
            token = inputTokens[pos + 1];
        }

        context->inference->setBatchSize(1);

        printf("\n🤖 Assistant\n");
        std::string answer;
        while (pos < seqLen) {
            int prevToken = token;

            context->inference->setPosition(pos);
            context->inference->setToken(0, token);

            context->inference->forward();

            token = context->sampler->sample(context->inference->logitsPipe);

            char *piece = context->tokenizer->decode(prevToken, token);
            bool isSafe = isSafePiece(piece);
            EosDetectorType eosType = eosDetector.append(token, isSafe ? piece : "");
            if (eosType == NOT_EOS || eosType == EOS) {
                char *delta = eosDetector.getDelta();
                if (delta != NULL) {
                    printf("%s", delta);
                    fflush(stdout);
                }
                eosDetector.clear();
            }
            pos++;
            if (eosType == EOS) break;
        }

        deltaItems.clear();
    } while (pos < seqLen);

    printf("(end of context)\n");
}

int main(int argc, char **argv) {
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        if (std::strcmp(args.mode, "inference") == 0)
            runInferenceApp(&args, &inference);
        else if (std::strcmp(args.mode, "chat") == 0)
            runInferenceApp(&args, &chat);
        else if (std::strcmp(args.mode, "worker") == 0)
            runWorkerApp(&args);
        else
            throw std::runtime_error("Unsupported mode");
    } catch (std::exception &e) {
        printf("🚨 Critical error: %s\n", e.what());
        return 1;
    }
    return 0;
}
