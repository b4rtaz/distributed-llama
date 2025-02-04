#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#include "llm.hpp"
#include "tokenizer.hpp"
#include "app.hpp"
#include <chrono>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    Timer() {
        startTime = std::chrono::high_resolution_clock::now();
    }
    NnSize elapsed() {
        auto endTime = std::chrono::high_resolution_clock::now();
        return (NnSize)std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    }
};

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
    for (;;) {
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
    }
    NnSize evalTime = evalTimer.elapsed();

    printf("🔷️ E %4u ms %s\n", evalTime, context->args->prompt);
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
            printf("🔶 P %4u ms %s\n", tokenTimer.elapsed(), piece);
            fflush(stdout);
        }
    }
    NnSize predTime = predTimer.elapsed();

    NnSize nEvalTokens = nInputTokens - 1;
    NnSize nPredTokens = pos - nEvalTokens;
    printf("\n");
    printf("Evaluation\n");
    printf("|   nBatches: %d\n", context->args->nBatches);
    printf("|    nTokens: %d\n", nEvalTokens);
    printf("|   tokens/s: %3.2f (%3.2f ms / 1 token)\n",
        nEvalTokens / (evalTime / 1000.0),
        evalTime / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("|    nTokens: %d\n", nPredTokens);
    printf("|   tokens/s: %3.2f (%3.2f ms / 1 token)\n",
        nPredTokens / (predTime / 1000.0),
        predTime / ((float) nPredTokens));
}

int main(int argc, char **argv) {
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, true);
        if (std::strcmp(args.mode, "inference") == 0)
            runInferenceApp(&args, &inference);
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
