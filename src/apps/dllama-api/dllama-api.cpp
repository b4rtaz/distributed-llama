#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <vector>

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
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    json parsedJson;
    HttpMethod method;
    std::string getMethod() {
        switch(method) {
            case HttpMethod::METHOD_GET:
                return "GET";
            case HttpMethod::METHOD_POST:
                return "POST";
            case HttpMethod::METHOD_PUT:
                return "PUT";
            case HttpMethod::METHOD_DELETE:
                return "DELETE";
            case HttpMethod::METHOD_UNKNOWN:
            default:
                return "UNKNOWN";
        }
    }
};

class HttpParser {
public:
    static HttpRequest parseRequest(const std::string& request) {
        HttpRequest httpRequest;

        // Split request into lines
        std::istringstream iss(request);
        std::string line;
        std::getline(iss, line);

        // Parse request line
        std::istringstream lineStream(line);
        std::string methodStr, path;
        lineStream >> methodStr >> path;
        httpRequest.method = parseMethod(methodStr);
        httpRequest.path = path;

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
                httpRequest.headers[key] = value;
            }
        }

        // Parse body
        std::getline(iss, httpRequest.body, '\0');

        if (httpRequest.body.size() > 0) {
            // printf("body: %s\n", httpRequest.body.c_str());
            httpRequest.parsedJson = json::parse(httpRequest.body);
        }
        return httpRequest;
    }
private:
    static HttpMethod parseMethod(const std::string& method) {
        if (method == "GET") {
            return HttpMethod::METHOD_GET;
        } else if (method == "POST") {
            return HttpMethod::METHOD_POST;
        } else if (method == "PUT") {
            return HttpMethod::METHOD_PUT;
        } else if (method == "DELETE") {
            return HttpMethod::METHOD_DELETE;
        } else {
            return HttpMethod::METHOD_UNKNOWN;
        }
    }
};

struct Route {
    std::string path;
    HttpMethod method;
    std::function<void(Socket&, HttpRequest&)> handler;
};

class Router {
public:
    static void routeRequest(Socket& client_socket, HttpRequest& request, std::vector<Route>& routes) {
        for (const auto& route : routes) {
            if (request.method == route.method && request.path == route.path) {
                route.handler(client_socket, request);
                return;
            }
        }
        notFoundHandler(client_socket);
    }
private:
    static void notFoundHandler(Socket& client_socket) {
        const char* data = "HTTP/1.1 404 Not Found\r\n";
        client_socket.write(data, strlen(data));
    }
};

/*
Generally speaking, the tokenizer.config.json would contain the chat template for the model
and depending on the model used you could set the chat template to follow
could possibly just for simplicity set this in ServerArgs with --chat-template
for this code draft I am assuming the use of llama 3 instruct
*/
std::string buildChatPrompt(Tokenizer *tokenizer, const std::vector<ChatMessage> &messages){
    std::ostringstream oss;
    for (const auto& message : messages) {
        oss << "<|start_header_id|>" << message.role << "<|end_header_id|>\n\n" << message.content << "<|eot_id|>";
    }

    oss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return oss.str();
}

void writeJsonResponse(Socket& socket, std::string json) {
    std::ostringstream buffer;
    buffer << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Content-Length: " << json.length() << "\r\n\r\n" << json;
    std::string data = buffer.str();
    socket.write(data.c_str(), data.size());
}

void writeStreamStartChunk(Socket& socket) {
    std::ostringstream buffer;
    buffer << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: text/event-stream; charset=utf-8\r\n"
        << "Connection: close\r\n"
        << "Transfer-Encoding: chunked\r\n\r\n";
    std::string data = buffer.str();
    socket.write(data.c_str(), data.size());
}

void writeStreamChunk(Socket& socket, const std::string data) {
    std::ostringstream buffer;
    buffer << std::hex << data.size() << "\r\n" << data << "\r\n";
    std::string d = buffer.str();
    socket.write(d.c_str(), d.size());
}

void writeStreamEndChunk(Socket& socket) {
    const char* endChunk = "0000\r\n\r\n";
    socket.write(endChunk, strlen(endChunk));
}

void writeChatCompletionChunk(Socket &socket, const std::string &delta, const bool stop){
    ChunkChoice choice;
    if (stop) {
        choice.finish_reason = "stop";
    } else {
        choice.delta = ChatMessageDelta("assistant", delta);
    }
    ChatCompletionChunk chunk = ChatCompletionChunk(choice);

    std::ostringstream buffer;
    buffer << "data: " << ((json)chunk).dump() << "\r\n\r\n";
    writeStreamChunk(socket, buffer.str());

    if (stop) {
        writeStreamChunk(socket, "data: [DONE]");
        writeStreamEndChunk(socket);
    }
}

void handleCompletionsRequest(Socket& socket, HttpRequest& request, Inference* inference, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec) {
    InferenceParams params;
    params.temperature = args->temperature;
    params.top_p = args->topp;
    params.seed = args->seed;
    params.stream = false;
    params.prompt = buildChatPrompt(tokenizer, parseChatMessages(request.parsedJson["messages"]));
    params.max_tokens = spec->seqLen - params.prompt.size();

    if (request.parsedJson.contains("stream")) {
        params.stream = request.parsedJson["stream"].get<bool>();
    }
    if (request.parsedJson.contains("temperature")) {
        params.temperature = request.parsedJson["temperature"].template get<float>();
        assert(params.temperature >= 0.0f);
        sampler->setTemp(params.temperature);
    }
    if (request.parsedJson.contains("seed")) {
        params.seed = request.parsedJson["seed"].template get<unsigned long long>();
        sampler->setSeed(params.seed);
    }
    if (request.parsedJson.contains("max_tokens")) {
        params.max_tokens = request.parsedJson["max_tokens"].template get<int>();
        assert(params.max_tokens <= spec->seqLen); //until rope scaling or similiar is implemented
    }
    if (request.parsedJson.contains("stop")) {
        params.stop = request.parsedJson["stop"].template get<std::vector<std::string>>();
    } else {
        const std::string defaultStop = "<|eot_id|>";
        params.stop = std::vector<std::string>{defaultStop};
    }

    printf("🔸");
    fflush(stdout);

    //Process the chat completion request
    std::vector<std::string> generated;
    generated.get_allocator().allocate(params.max_tokens);

    if (params.stream) {
        writeStreamStartChunk(socket);
    }

    int promptLength = params.prompt.length();
    int nPromptTokens;
    int promptTokens[promptLength + 3];
    char prompt[promptLength + 1];
    prompt[promptLength] = 0;
    strcpy(prompt, params.prompt.c_str());
    tokenizer->encode(prompt, promptTokens, &nPromptTokens, true, false);

    int token = promptTokens[0];
    pos_t maxPos = nPromptTokens + params.max_tokens;
    if (maxPos > spec->seqLen) maxPos = spec->seqLen;
    bool eosEncountered = false;
    for (pos_t pos = 0; pos < maxPos; pos++) {
        float* logits = inference->infer(token, pos);

        if (pos < nPromptTokens - 1) {
            token = promptTokens[pos + 1];
        } else {
            int prevToken = token;
            token = sampler->sample(logits);

            if (token == tokenizer->eosId) eosEncountered = true;

            char* piece = tokenizer->decode(prevToken, token);

            bool safePiece = isSafePiece(piece);
            
            if (!params.stop.empty() && safePiece) {
                std::string concatenatedTokens;
                int startIndex = std::max(0, static_cast<int>(generated.size()) - 7);
                for (int i = startIndex; i < generated.size(); ++i) {
                    concatenatedTokens += generated[i];
                }
                concatenatedTokens += std::string(piece);

                for (const auto& word : params.stop) {
                    if (concatenatedTokens.find(word) != std::string::npos) {
                        eosEncountered = true;
                        break;
                    }
                }
            }

            if (eosEncountered) break;

            std::string string = std::string(piece);
            safePrintf(piece);
            fflush(stdout);

            generated.push_back(string);
            if (params.stream) {
                writeChatCompletionChunk(socket, string, false);
            }
        }
    }

    if (!params.stream) {
        ChatMessage chatMessage = ChatMessage("assistant", std::accumulate(generated.begin(), generated.end(), std::string("")));
        Choice responseChoice = Choice(chatMessage);
        ChatCompletion completion = ChatCompletion(responseChoice);
        completion.usage = ChatUsage(nPromptTokens, generated.size(), nPromptTokens + generated.size());

        std::string chatJson = ((json)completion).dump();
        writeJsonResponse(socket, chatJson);
    } else {
        writeChatCompletionChunk(socket, "", true);
    }
    printf("🔶\n");
    fflush(stdout);
}

void handleModelsRequest(Socket& client_socket, HttpRequest& request) {
    writeJsonResponse(client_socket,
        "{ \"object\": \"list\","
        "\"data\": ["
        "{ \"id\": \"dl\", \"object\": \"model\", \"created\": 0, \"owned_by\": \"user\" }"
        "] }");
}

void server(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, AppArgs* args, TransformerSpec* spec) {
    SocketServer* server = new SocketServer(args->port);
    printf("Server URL: http://127.0.0.1:%d/v1/\n", args->port);

    std::vector<Route> routes = {
        {
            "/v1/chat/completions",
            HttpMethod::METHOD_POST,
            std::bind(&handleCompletionsRequest, std::placeholders::_1, std::placeholders::_2, inference, tokenizer, sampler, args, spec)
        },
        {
            "/v1/models",
            HttpMethod::METHOD_GET,
            std::bind(&handleModelsRequest, std::placeholders::_1, std::placeholders::_2)
        }
    };

    while (true) {
        try {
            // Accept incoming connection
            Socket client = server->accept();
            // Read the HTTP request
            std::vector<char> httpRequest = client.readHttpRequest();
            // Parse the HTTP request
            HttpRequest request = HttpParser::parseRequest(std::string(httpRequest.begin(), httpRequest.end()));
            // Handle the HTTP request
            printf("🔷 %s %s\n", request.getMethod().c_str(), request.path.c_str());
            Router::routeRequest(client, request, routes);
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
