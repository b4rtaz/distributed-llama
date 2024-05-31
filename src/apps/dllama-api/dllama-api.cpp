#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

#include "types.hpp"
#include "../../utils.hpp"
#include "../../socket.hpp"
#include "../../transformer.hpp"
#include "../../tokenizer.hpp"
#include "../../app.hpp"
#include "../../common/json.hpp"

using json = nlohmann::json;

enum class HttpMethod {
    METHOD_GET = 0,
    METHOD_POST = 1,
    METHOD_PUT = 2,
    METHOD_DELETE = 3,
    METHOD_UNKNOWN = 4
};

class HttpRequest {
public:
    static HttpRequest read(Socket& socket) {
        HttpRequest req(&socket);

        std::vector<char> httpRequest = socket.readHttpRequest();
        // Parse the HTTP request
        std::string data = std::string(httpRequest.begin(), httpRequest.end());

        // Split request into lines
        std::istringstream iss(data);
        std::string line;
        std::getline(iss, line);

        // Parse request line
        std::istringstream lineStream(line);
        std::string methodStr, path;
        lineStream >> methodStr >> path;
        req.method = parseMethod(methodStr);
        req.path = path;

        // Parse headers
        while (std::getline(iss, line) && line != "\r") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2); // Skip ': ' after key
                // Trim whitespace and non-printable characters from header value
                value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                    return std::isspace(c) || !std::isprint(c);
                }), value.end());
                req.headers[key] = value;
            }
        }

        // Parse body
        std::getline(iss, req.body, '\0');

        if (req.body.size() > 0) {
            // printf("body: %s\n", req.body.c_str());
            req.parsedJson = json::parse(req.body);
        }
        return req;
    }

    static HttpMethod parseMethod(const std::string& method) {
        if (method == "GET") return HttpMethod::METHOD_GET;
        if (method == "POST") return HttpMethod::METHOD_POST;
        if (method == "PUT") return HttpMethod::METHOD_PUT;
        if (method == "DELETE") return HttpMethod::METHOD_DELETE;
        return HttpMethod::METHOD_UNKNOWN;
    }

private:
    Socket* socket;
public:
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    json parsedJson;
    HttpMethod method;

    HttpRequest(Socket* socket) {
        this->socket = socket;
    }

    std::string getMethod() {
        if (method == HttpMethod::METHOD_GET) return "GET";
        if (method == HttpMethod::METHOD_POST) return "POST";
        if (method == HttpMethod::METHOD_PUT) return "PUT";
        if (method == HttpMethod::METHOD_DELETE) return "DELETE";
        return "UNKNOWN";
    }

    void writeNotFound() {
        const char* data = "HTTP/1.1 404 Not Found\r\n";
        socket->write(data, strlen(data));
    }

    void writeJson(std::string json) {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Content-Length: " << json.length() << "\r\n\r\n" << json;
        std::string data = buffer.str();
        socket->write(data.c_str(), data.size());
    }

    void writeStreamStartChunk() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: text/event-stream; charset=utf-8\r\n"
            << "Connection: close\r\n"
            << "Transfer-Encoding: chunked\r\n\r\n";
        std::string data = buffer.str();
        socket->write(data.c_str(), data.size());
    }

    void writeStreamChunk(const std::string data) {
        std::ostringstream buffer;
        buffer << std::hex << data.size() << "\r\n" << data << "\r\n";
        std::string d = buffer.str();
        socket->write(d.c_str(), d.size());
    }

    void writeStreamEndChunk() {
        const char* endChunk = "0000\r\n\r\n";
        socket->write(endChunk, strlen(endChunk));
    }
};

struct Route {
    std::string path;
    HttpMethod method;
    std::function<void(HttpRequest&)> handler;
};

class Router {
public:
    static void resolve(HttpRequest& request, std::vector<Route>& routes) {
        for (const auto& route : routes) {
            if (request.method == route.method && request.path == route.path) {
                route.handler(request);
                return;
            }
        }
        request.writeNotFound();
    }
};

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
        request.writeStreamChunk("data: [DONE]");
        request.writeStreamEndChunk();
    }
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
    Inference* inference;
    Tokenizer* tokenizer;
    Sampler* sampler;
    AppArgs* args;
    TransformerSpec* spec;
    EosDetector* eosDetector;
    NaiveCache naiveCache;

public:
    ApiServer( Inference* inference, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec, EosDetector* eosDetector) {
        this->inference = inference;
        this->tokenizer = tokenizer;
        this->sampler = sampler;
        this->args = args;
        this->spec = spec;
        this->eosDetector = eosDetector;
    }

    std::string buildChatPrompt(std::vector<ChatMessage> messages) {
        assert(tokenizer->nChatTemplates == 6);

        std::ostringstream buffer;
        for (const auto& message : messages) {
            buffer << tokenizer->chatTemplate[0]; // chatMessageStart
            buffer << tokenizer->chatTemplate[1]; // chatRoleStart
            buffer << message.role;
            buffer << tokenizer->chatTemplate[2]; // chatRoleEnd
            buffer << message.content;
            buffer << tokenizer->chatTemplate[3]; // chatMessageEnd
        }

        buffer << tokenizer->chatTemplate[4]; // chatGenerationPrompt
        return buffer.str();
    }

    void complete(HttpRequest& request) {
        InferenceParams params = parseRequest(request);

        pos_t startPos = 0;
        std::vector<ChatMessage> deltaPrompt = params.messages;
        naiveCache.resolveDeltaPrompt(deltaPrompt, startPos);

        printf("🔸");
        fflush(stdout);

        std::string inputPrompt = buildChatPrompt(deltaPrompt);
        int promptLength = inputPrompt.size();
        int nPromptTokens;
        int promptTokens[promptLength + 3];
        char prompt[promptLength + 1];
        prompt[promptLength] = 0;
        strcpy(prompt, inputPrompt.c_str());
        tokenizer->encode(prompt, promptTokens, &nPromptTokens, true, false);
        int promptEndPos = startPos + nPromptTokens;

        for (size_t j = 0; j < deltaPrompt.size(); j++) {
            naiveCache.push(NaiveCacheItem(promptEndPos, deltaPrompt[j]));
        }

        pos_t maxPos = params.max_tokens > 0 ? (promptEndPos + params.max_tokens) : spec->seqLen;
        if (maxPos > spec->seqLen) maxPos = spec->seqLen;

        if (params.stream) {
            request.writeStreamStartChunk();
        }

        std::string buffer;
        size_t nStops = params.stop.size();

        int token = promptTokens[0];
        pos_t pos = startPos;
        for (; pos < maxPos; pos++) {
            float* logits = inference->infer(token, pos);

            if (pos < promptEndPos - 1) {
                token = promptTokens[pos - startPos + 1];
            } else {
                int prevToken = token;
                token = sampler->sample(logits);

                char* piece = tokenizer->decode(prevToken, token);
                bool isSafe = isSafePiece(piece);

                int eosType = eosDetector->append(token, isSafe ? piece : "");

                if (isSafePiece(piece)) {
                    printf("%s", piece);
                    fflush(stdout);
                }

                if (eosType == NOT_EOS || eosType == EOS) {
                    char* delta = eosDetector->getDelta();
                    if (delta != NULL) {
                        std::string deltaStr(delta);
                        if (params.stream)
                            writeChatCompletionChunk(request, deltaStr, false);
                        buffer += deltaStr;
                    }
                    eosDetector->clear();
                }
                if (eosType == EOS) break;
            }
        }

        ChatMessage chatMessage("assistant", buffer);
        if (pos == spec->seqLen) {
            naiveCache.clear();
        } else {
            naiveCache.push(NaiveCacheItem(pos, chatMessage));
        }

        if (params.stream) {
            writeChatCompletionChunk(request, "", true);
        } else {
            int nCompletionTokens = pos - promptEndPos;
            ChatUsage usage(nPromptTokens, nCompletionTokens, nPromptTokens + nCompletionTokens);
            Choice choice(chatMessage);
            ChatCompletion completion(choice, usage);
            std::string chatJson = ((json)completion).dump();
            request.writeJson(chatJson);
        }
        printf("🔶\n");
        fflush(stdout);
    }

private:
    InferenceParams parseRequest(HttpRequest& request) {
        InferenceParams params;
        params.temperature = args->temperature;
        params.top_p = args->topp;
        params.seed = args->seed;
        params.stream = false;
        params.messages = parseChatMessages(request.parsedJson["messages"]);
        params.max_tokens = -1;

        if (request.parsedJson.contains("stream")) {
            params.stream = request.parsedJson["stream"].get<bool>();
        }
        if (request.parsedJson.contains("temperature")) {
            params.temperature = request.parsedJson["temperature"].template get<float>();
        }
        if (request.parsedJson.contains("seed")) {
            params.seed = request.parsedJson["seed"].template get<unsigned long long>();
            sampler->setSeed(params.seed);
        }
        if (request.parsedJson.contains("max_tokens")) {
            params.max_tokens = request.parsedJson["max_tokens"].template get<int>();
        }
        if (request.parsedJson.contains("stop")) {
            params.stop = request.parsedJson["stop"].template get<std::vector<std::string>>();
        } else {
            const std::string defaultStop = "<|eot_id|>";
            params.stop = std::vector<std::string>{defaultStop};
        }
        return params;
    }
};

void handleCompletionsRequest(HttpRequest& request, ApiServer* api) {
    api->complete(request);
}

void handleModelsRequest(HttpRequest& request) {
    request.writeJson(
        "{ \"object\": \"list\","
        "\"data\": ["
        "{ \"id\": \"dl\", \"object\": \"model\", \"created\": 0, \"owned_by\": \"user\" }"
        "] }");
}

void server(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, AppArgs* args, TransformerSpec* spec) {
    if (tokenizer->chatEosId < 0) {
        printf("⛔ 0.8.0 version introduced a new format of the tokenizer that includes chatEosId. Please update your tokenizer.\n");
        exit(EXIT_FAILURE);
    }

    SocketServer* server = new SocketServer(args->port);

    const bool hasExtraStop = tokenizer->chatTemplate[5][0] != '\0';
    const int nStops = hasExtraStop ? 2 : 1;
    char* stops[nStops];
    stops[0] = tokenizer->vocab[tokenizer->chatEosId];
    if (hasExtraStop)
        stops[1] = tokenizer->chatTemplate[5];

    EosDetector eosDetector(tokenizer->chatEosId, nStops, (const char**)stops, 1, 1);
    ApiServer api(inference, tokenizer, sampler, args, spec, &eosDetector);

    printf("Server URL: http://127.0.0.1:%d/v1/\n", args->port);

    std::vector<Route> routes = {
        {
            "/v1/chat/completions",
            HttpMethod::METHOD_POST,
            std::bind(&handleCompletionsRequest, std::placeholders::_1, &api)
        },
        {
            "/v1/models",
            HttpMethod::METHOD_GET,
            std::bind(&handleModelsRequest, std::placeholders::_1)
        }
    };

    while (true) {
        try {
            Socket client = server->accept();
            HttpRequest request = HttpRequest::read(client);
            printf("🔷 %s %s\n", request.getMethod().c_str(), request.path.c_str());
            Router::resolve(request, routes);
        } catch (ReadSocketException& ex) {
            printf("Read socket error: %d %s\n", ex.code, ex.message);
        } catch (WriteSocketException& ex) {
            printf("Write socket error: %d %s\n", ex.code, ex.message);
        }
    }

    delete server;
}

int main(int argc, char *argv[]) {
    initQuants();

    AppArgs args = AppArgs::parse(argc, argv, false);
    App::run(&args, server);
    return EXIT_SUCCESS;
}
