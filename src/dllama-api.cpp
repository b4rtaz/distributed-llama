#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#include "tokenizer.hpp"
#include "app.hpp"
#include "http.hpp"
#include "json.hpp"
#include "dllama-api.hpp"
#include "nn/nn-network.hpp"

typedef unsigned int pos_t;

using json = nlohmann::json;

void writeChatCompletionChunk(HttpRequest &request, const std::string &delta, const bool stop){
    ChunkChoice choice;
    if (stop) {
        choice.finish_reason = "stop";
    } else {
        choice.delta = ChatMessageDelta("assistant", delta);
    }
    ChatCompletionChunk chunk = ChatCompletionChunk(choice);

    std::ostringstream buffer;
    buffer << "data: " << ((json)chunk).dump() << "\r\n\r\n";
    request.writeStreamChunk(buffer.str());

    if (stop) {
        request.writeStreamChunk("data: [DONE]\r\n\r\n");
        request.writeStreamEndChunk();
    }
}

static bool tryParseJsonObject(const std::string &text, json &out) {
    json parsed = json::parse(text, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object())
        return false;
    out = std::move(parsed);
    return true;
}

static bool tryFindJsonFragmentAtEnd(const std::string &text, json &out) {
    if (tryParseJsonObject(text, out))
        return true;

    bool inString = false;
    int depth = 0;
    size_t end = std::string::npos;

    for (size_t i = text.size(); i-- > 0;) {
        char c = text[i];
        if (c == '"') {
            size_t backslashes = 0;
            for (size_t j = i; j > 0 && text[j - 1] == '\\'; j--)
                backslashes++;
            if ((backslashes % 2) == 0)
                inString = !inString;
            continue;
        }
        if (inString)
            continue;
        if (c == '}') {
            if (end == std::string::npos)
                end = i;
            depth++;
            continue;
        }
        if (c == '{') {
            if (depth == 0)
                continue;
            depth--;
            if (depth == 0 && end != std::string::npos) {
                std::string candidate = text.substr(i, end - i + 1);
                if (tryParseJsonObject(candidate, out))
                    return true;
                end = std::string::npos;
            }
        }
    }
    return false;
}

static std::string tryBuildToolsSystemPrompt(const std::vector<Tool> &tools, const ToolChoice &choice) {
    json toolJson = tools;
    std::string prompt = "You have access to the following tools:\n";
    prompt += toolJson.dump();
    prompt += "\n\nWhen you decide to call a tool, respond with a JSON object in this format:\n";
    prompt += "{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"tool_name\",\"arguments\":\"{...}\"}}]}";
    if (choice.kind == TOOL_CHOICE_NONE)
        prompt += "\nDo not call any tools.";
    else if (choice.kind == TOOL_CHOICE_REQUIRED)
        prompt += "\nYou must call a tool.";
    else if (choice.kind == TOOL_CHOICE_NAMED)
        prompt += "\nYou must call the tool named: " + choice.tool_name + ".";
    return prompt;
}

class NaiveCacheItem {
public:
    pos_t endPos;
    ChatMessage message;
    NaiveCacheItem(pos_t endPos, ChatMessage message) {
        this->endPos = endPos;
        this->message = message;
    }
};

class NaiveCache {
private:
    std::vector<NaiveCacheItem> cache;
public:
    void push(NaiveCacheItem item) {
        cache.push_back(item);
    }

    void clear() {
        cache.clear();
    }

    bool resolveDeltaPrompt(std::vector<ChatMessage>& messages, pos_t& startPos) {
        size_t cacheSize = cache.size();
        if (cacheSize == 0)
            return false;
        if (messages.size() > cacheSize) {
            size_t i = 0;
            while (i < cacheSize) {
                if (
                    cache[i].message.role != messages[i].role ||
                    cache[i].message.content != messages[i].content
                ) break;
                i++;
            }
            if (i == cacheSize) {
                startPos = cache[i - 1].endPos;
                messages.erase(messages.begin(), messages.begin() + i);
                printf("🐤 Found naive cache for %zu messages, pos=%d\n", i, startPos);
                return true;
            }
        }
        cache.clear();
        return false;
    }
};

class ApiServer {
private:
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    Sampler *sampler;
    AppCliArgs *args;
    LlmHeader *header;
    EosDetector *eosDetector;
    ChatTemplateGenerator *templateGenerator;
    NaiveCache naiveCache;

public:
    ApiServer(RootLlmInference *inference, Tokenizer *tokenizer, Sampler *sampler, AppCliArgs *args, LlmHeader *header, EosDetector *eosDetector, ChatTemplateGenerator *templateGenerator) {
        this->inference = inference;
        this->tokenizer = tokenizer;
        this->sampler = sampler;
        this->args = args;
        this->header = header;
        this->eosDetector = eosDetector;
        this->templateGenerator = templateGenerator;
    }

    void complete(HttpRequest& request) {
        InferenceParams params = parseRequest(request);

        pos_t startPos = 0;
        std::vector<ChatMessage> deltaPrompt = params.messages;
        naiveCache.resolveDeltaPrompt(deltaPrompt, startPos);

        size_t nInputItems = deltaPrompt.size();
        std::unique_ptr<ChatItem[]> inputItemsPtr(new ChatItem[nInputItems]);
        ChatItem *inputItems = inputItemsPtr.get();
        for (size_t i = 0; i < nInputItems; i++) {
            inputItems[i].role = deltaPrompt[i].role;
            inputItems[i].message = deltaPrompt[i].content;
        }

        GeneratedChat inputPrompt = templateGenerator->generate(nInputItems, inputItems, true);
        printf("🔹\033[34m%s", inputPrompt.content);
        fflush(stdout);

        int nPromptTokens;
        std::unique_ptr<int[]> promptTokensPtr(new int[inputPrompt.length + 2]);
        int *promptTokens = promptTokensPtr.get();
        bool isStart = startPos == 0;
        tokenizer->encode((char*)inputPrompt.content, promptTokens, &nPromptTokens, isStart, true);

        pos_t promptEndPos = startPos + nPromptTokens - 1;
        if (promptEndPos > header->seqLen)
            promptEndPos = header->seqLen;

        pos_t maxPredPos = params.max_tokens > 0 ? (promptEndPos + params.max_tokens) : header->seqLen;
        if (maxPredPos > header->seqLen)
            maxPredPos = header->seqLen;

        for (size_t j = 0; j < deltaPrompt.size(); j++) {
            naiveCache.push(NaiveCacheItem(promptEndPos, deltaPrompt[j]));
        }

        std::string buffer;

        if (params.stream)
            request.writeStreamStartChunk();
        if (inputPrompt.publicPrompt != nullptr) {
            if (params.stream)
                writeChatCompletionChunk(request, inputPrompt.publicPrompt, false);
            buffer += inputPrompt.publicPrompt;
        }

        printf("(%d tokens)\033\n[0m🔸", promptEndPos - startPos);
        fflush(stdout);

        NnUint pos = startPos;
        int token;
        for (NnUint i = 0; ;) {
            long remainingTokens = promptEndPos - pos;
            if (remainingTokens <= 0)
                break;

            NnUint batchSize = remainingTokens < args->nBatches
                ? remainingTokens
                : args->nBatches;

            inference->setBatchSize(batchSize);
            inference->setPosition(pos);
            for (NnUint j = 0; j < batchSize; j++)
                inference->setToken(j, promptTokens[i + j]);

            inference->forward();

            i += batchSize;
            pos += batchSize;
            token = promptTokens[i];
        }

        inference->setBatchSize(1);
        tokenizer->resetDecoder();
        eosDetector->reset();

        for (; pos < maxPredPos;) {
            inference->setPosition(pos);
            inference->setToken(0, token);
            inference->forward();

            token = sampler->sample(inference->outputPipe);

            char *piece = tokenizer->decode(token);
            EosDetectorType eosType = eosDetector->append(token, piece);

            if (piece != nullptr) {
                printf("%s", piece);
                fflush(stdout);
            }

            if (eosType == NOT_EOS || eosType == EOS) {
                char *delta = eosDetector->getDelta();
                if (delta != nullptr) {
                    std::string deltaStr(delta);
                    if (params.stream)
                        writeChatCompletionChunk(request, deltaStr, false);
                    buffer += deltaStr;
                }
                eosDetector->reset();
            }
            pos++;
            if (eosType == EOS) break;
        }

        ChatMessage reply("assistant", buffer);
        if (pos == header->seqLen) {
            naiveCache.clear();
        } else {
            naiveCache.push(NaiveCacheItem(pos, reply));
        }

        if (params.stream) {
            writeChatCompletionChunk(request, "", true);
        } else {
            Choice choice(reply);

            if (!params.tools.empty()) {
                std::vector<ToolCall> parsedToolCalls;
                json tempJson;
                if (tryFindJsonFragmentAtEnd(buffer, tempJson) &&
                    tryParseToolCallsFromJson(tempJson, parsedToolCalls)) {
                    choice.message.tool_calls = parsedToolCalls;
                    choice.finish_reason = "tool_calls";
                }
            }

            int nCompletionTokens = pos - promptEndPos;
            ChatUsage usage(nPromptTokens, nCompletionTokens, nPromptTokens + nCompletionTokens);
            ChatCompletion completion(choice, usage);
            std::string chatJson = ((json)completion).dump();
            request.writeJson(chatJson);
        }
        printf("🔶\n");
        fflush(stdout);
    }

private:
    InferenceParams parseRequest(HttpRequest& request) {
        InferenceParams params = parseInferenceParams(
            request.parsedJson,
            args->temperature,
            args->topp,
            args->seed);
        sampler->setSeed(params.seed);
        if (!params.tools.empty()) {
            std::string prompt = tryBuildToolsSystemPrompt(params.tools, params.tool_choice);
            ChatMessage toolMessage("system", prompt);
            params.messages.insert(params.messages.begin(), toolMessage);
        }
        return params;
    }
};

void handleCompletionsRequest(HttpRequest& request, ApiServer *api) {
    api->complete(request);
}

void handleModelsRequest(HttpRequest& request, const char* modelPath) {
    Model model(getFileName(modelPath));
    ModelList list(model);
    std::string response = ((json)list).dump();
    request.writeJson(response);
}

static void server(AppInferenceContext *context) {
    TokenizerChatStops stops(context->tokenizer);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    EosDetector eosDetector(stops.nStops, context->tokenizer->eosTokenIds.data(), stops.stops, stops.maxStopLength, stops.maxStopLength);
    ApiServer api(context->inference, context->tokenizer, context->sampler, context->args, context->header, &eosDetector, &templateGenerator);

    if (strcmp(context->args->host, "0.0.0.0") == 0 ||
        strcmp(context->args->host, "127.0.0.1") == 0)
        printf("Server URL: http://localhost:%d/v1/\n", context->args->port);

    std::vector<Route> routes = {
        {
            "/v1/chat/completions",
            HttpMethod::METHOD_POST,
            std::bind(&handleCompletionsRequest, std::placeholders::_1, &api)
        },
        {
            "/v1/models",
            HttpMethod::METHOD_GET,
            std::bind(&handleModelsRequest, std::placeholders::_1, context->args->modelPath)
        }
    };

    runHttpServer(context->args->host, context->args->port, routes);
}

#ifdef _WIN32
    #define EXECUTABLE_NAME "dllama-api.exe"
#else
    #define EXECUTABLE_NAME "dllama-api"
#endif

void usage() {
    fprintf(stderr, "Usage: %s {--model <path>} {--tokenizer <path>} [--host <addr>] [--port <p>]\n", EXECUTABLE_NAME);
    fprintf(stderr, "        [--buffer-float-type {f32|f16|q40|q80}]\n");
    fprintf(stderr, "        [--weights-float-type {f32|f16|q40|q80}]\n");
    fprintf(stderr, "        [--max-seq-len <max>]\n");
    fprintf(stderr, "        [--nthreads <n>]\n");
    fprintf(stderr, "        [--workers <ip:port> ...]\n");
    fprintf(stderr, "        [--temperature <temp>]\n");
    fprintf(stderr, "        [--topp <t>]\n");
    fprintf(stderr, "        [--seed <s>]\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  sudo nice -n -20 ./dllama-api --port 9990 --nthreads 4 \\\n");
    fprintf(stderr, "    --model dllama_model_llama3_2_3b_instruct_q40.m \\\n");
    fprintf(stderr, "    --tokenizer dllama_tokenizer_llama3_2_3b_instruct_q40.t \\\n");
    fprintf(stderr, "    --buffer-float-type q80 --max-seq-len 8192 \\\n");
    fprintf(stderr, "    --workers 10.0.0.2:9998 10.0.0.3:9998 10.0.0.4:9998\n");
    fflush(stderr);
}

int main(int argc, char *argv[]) {
    AppCliArgs args = AppCliArgs::parse(argc, argv, false);
    if (args.help) {
        usage();
        return EXIT_SUCCESS;
    }

    initQuants();
    initSockets();

    while (true) {
        try {
            runInferenceApp(&args, server);
        } catch (const NnConnectionSocketException &e) {
            printf("🚨 Connection error: %s\n", e.what());
        } catch (const NnExecutorException &e) {
            printf("🚨 Inference error: %s\n", e.what());
        }

        printf("🔄 Retrying in 3 seconds...\n");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        args.info = false;
    }

    cleanupSockets();
    return EXIT_SUCCESS;
}
