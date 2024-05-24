#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <iostream>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

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

        // Parse JSON if Content-Type is application/json
        httpRequest.parsedJson = json::object();
        if(httpRequest.headers.find("Content-Type") != httpRequest.headers.end()){
            if(httpRequest.headers["Content-Type"] == "application/json"){
                httpRequest.parsedJson = json::parse(httpRequest.body);
            }
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
        std::string header = "HTTP/1.1 404 Not Found\r\n";
        client_socket.write(header.c_str(), header.length());
    }
};

struct ChatMessageDelta {
    std::string role;
    std::string content;

    ChatMessageDelta() : role(""), content("") {}
    ChatMessageDelta(const std::string& role_, const std::string& content_) : role(role_), content(content_) {}
};

// Define to_json for Delta struct
void to_json(json& j, const ChatMessageDelta& msg) {
    j = json{{"role", msg.role}, {"content", msg.content}};
}

struct ChatMessage {
    std::string role;
    std::string content;

    ChatMessage() : role(""), content("") {}
    ChatMessage(const std::string& role_, const std::string& content_) : role(role_), content(content_) {}
};

// Define to_json for ChatMessage struct
void to_json(json& j, const ChatMessage& msg) {
    j = json{{"role", msg.role}, {"content", msg.content}};
}

struct ChunkChoice {
    int index;
    ChatMessageDelta delta;
    std::string finish_reason;

    ChunkChoice() : index(0) {}
};

// Define to_json for ChunkChoice struct
void to_json(json& j, const ChunkChoice& choice) {
    j = json{{"index", choice.index}, {"delta", choice.delta}, {"finish_reason", choice.finish_reason}};
}

struct Choice {
    int index;
    ChatMessage message;
    std::string finish_reason;

    Choice() : finish_reason("") {}
    Choice(ChatMessage &message_) : message(message_), finish_reason("") {}
    Choice(const std::string &reason_) : finish_reason(reason_) {}
};

// Define to_json for Choice struct
void to_json(json& j, const Choice& choice) {
    j = json{{"index", choice.index}, {"message", choice.message}, {"finish_reason", choice.finish_reason}};
}

struct ChatCompletionChunk {
    std::string id;
    std::string object;
    long long created;
    std::string model;
    std::vector<ChunkChoice> choices;

    ChatCompletionChunk(ChunkChoice &choice_) 
        : id("chatcmpl-test"), object("chat.completion"), model("Distributed Model") {
        created = std::time(nullptr); // Set created to current Unix timestamp
        choices.push_back(choice_);
    }
};

// Define to_json for ChatCompletionChunk struct
void to_json(json& j, const ChatCompletionChunk& completion) {
    j = json{{"id", completion.id},
             {"object", completion.object},
             {"created", completion.created},
             {"model", completion.model},
             {"choices", completion.choices}};
}

// Struct to represent the usage object
struct ChatUsage {
    int prompt_tokens;
    int completion_tokens;
    int total_tokens;
    ChatUsage() : prompt_tokens(0), completion_tokens(0), total_tokens(0) {}
    ChatUsage(int pt, int ct, int tt) : prompt_tokens(pt), completion_tokens(ct), total_tokens(tt) {}
};

// Struct to represent the chat completion object
struct ChatCompletion {
    std::string id;
    std::string object;
    long long created; // Unix timestamp
    std::string model;
    std::vector<Choice> choices;
    ChatUsage usage;

    ChatCompletion(Choice &choice_) 
        : id("chatcmpl-test"), object("chat.completion"), model("Distributed Model") {
        created = std::time(nullptr); // Set created to current Unix timestamp
        choices.push_back(choice_);
    }
};

// Define to_json for ChatCompletion struct
void to_json(json& j, const ChatCompletion& completion) {
    j = json{{"id", completion.id},
             {"object", completion.object},
             {"created", completion.created},
             {"model", completion.model},
             {"choices", completion.choices}};
}

struct InferenceParams {
    std::string prompt;
    int max_tokens;
    float temperature;
    float top_p;
    std::vector<std::string> stop;
    bool stream;
    unsigned long long seed;
};

std::vector<ChatMessage> parseChatMessages(json &json){
    std::vector<ChatMessage> messages;
    messages.reserve(json.size());

    for (const auto& item : json) {
        messages.emplace_back(
           item["role"].template get<std::string>(),
           item["content"].template get<std::string>()
        );
    }

    return messages;
}

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

void outputChatCompletionChunk(Socket &client_socket, const std::string &delta, const std::string &finish_reason){
    ChunkChoice choice;
    
    if(finish_reason.size() > 0){
        choice.finish_reason = finish_reason;
    }
    else{
        choice.delta = ChatMessageDelta("assistant", delta);
    }

    ChatCompletionChunk chunk = ChatCompletionChunk(choice);
    
    std::ostringstream oss;
    
    oss << "data: " << ((json)chunk).dump() << "\n\n";

    if(finish_reason.size() > 0){ 
        oss << "data: [DONE]\n\n";
    }

    std::string chunkResponse = oss.str();

    // Format the chunked response
    std::ostringstream formattedChunk;
    formattedChunk << std::hex << chunkResponse.length() << "\r\n" << chunkResponse << "\r\n";

    client_socket.write(formattedChunk.str().c_str(), formattedChunk.str().length());
}

void handleCompletionsRequest(Socket& client_socket, HttpRequest& request, Inference* inference, Tokenizer* tokenizer, Sampler* sampler, AppArgs* args, TransformerSpec* spec) {
    printf("Handling Completion Request\n");
    // Set inference arguments
    InferenceParams inferParams;
    inferParams.temperature = args->temperature;
    inferParams.top_p = args->topp;
    inferParams.seed = args->seed;
    inferParams.stream = false;
    inferParams.prompt = buildChatPrompt(tokenizer, parseChatMessages(request.parsedJson["messages"]));
    inferParams.max_tokens = spec->seqLen - inferParams.prompt.size();

    if(request.parsedJson.contains("stream")){
        inferParams.stream = request.parsedJson["stream"].template get<bool>();
    }
    if(request.parsedJson.contains("temperature")){
        inferParams.temperature = request.parsedJson["temperature"].template get<float>();
        assert(inferParams.temperature >= 0.0f);
        sampler->setTemp(inferParams.temperature);
    }
    if(request.parsedJson.contains("seed")){
        inferParams.seed = request.parsedJson["seed"].template get<unsigned long long>();
        sampler->setSeed(inferParams.seed);
    }
    if(request.parsedJson.contains("max_tokens")){
        inferParams.max_tokens = request.parsedJson["max_tokens"].template get<int>();
        assert(inferParams.max_tokens <= spec->seqLen); //until rope scaling or similiar is implemented
    }
    if(request.parsedJson.contains("stop")){
        inferParams.stop = request.parsedJson["stop"].template get<std::vector<std::string>>();
    }

    //Process the chat completion request
    std::vector<std::string> generated;
    generated.get_allocator().allocate(inferParams.max_tokens);

    if (inferParams.stream) {
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: text/event-stream; charset=utf-8\r\n"
            << "Connection: keep-alive\r\n"
            << "Transfer-Encoding: chunked\r\n\r\n";

        client_socket.write(oss.str().c_str(), oss.str().length());
    }

    int promptLength = inferParams.prompt.length();
    int nPromptTokens;
    int promptTokens[promptLength + 3];
    char prompt[promptLength + 1];
    prompt[promptLength] = 0;
    strcpy(prompt, inferParams.prompt.c_str());
    tokenizer->encode(prompt, promptTokens, &nPromptTokens, true, false);

    int token = promptTokens[0];
    pos_t maxPos = nPromptTokens + inferParams.max_tokens;
    if (maxPos > spec->seqLen) maxPos = spec->seqLen;
    bool eosEncountered = false;
    for (pos_t pos = 0; pos < maxPos; pos++) {
        float* logits = inference->infer(token, pos);

        if (pos < nPromptTokens - 1) {
            token = promptTokens[pos + 1];
        }
        else {
            int prevToken = token;
            token = sampler->sample(logits);

            if (token == tokenizer->eosId) eosEncountered = true;

            char* piece = tokenizer->decode(prevToken, token);

            bool safePiece = isSafePiece(piece);
            
            if (!inferParams.stop.empty() && safePiece) {
                std::string concatenatedTokens;
                int startIndex = std::max(0, static_cast<int>(generated.size()) - 7);
                for (int i = startIndex; i < generated.size(); ++i) {
                    concatenatedTokens += generated[i];
                }
                concatenatedTokens += std::string(piece);

                for (const auto& word : inferParams.stop) {
                    if (concatenatedTokens.find(word) != std::string::npos) {
                        eosEncountered = true;
                        break;
                    }
                }
            }

            if (eosEncountered) break;

            std::string string = std::string(piece);

            //char string[100];
            //strcpy(string, piece);
            safePrintf(piece);

            generated.push_back(string);
            
            if (inferParams.stream) {
                outputChatCompletionChunk(client_socket, string, "");
            }
        }
    }

    if (!inferParams.stream) {
        ChatMessage chatMessage = ChatMessage("assistant", std::accumulate(generated.begin(), generated.end(), std::string("")));
        Choice responseChoice = Choice(chatMessage);
        ChatCompletion completion = ChatCompletion(responseChoice);
        completion.usage = ChatUsage(nPromptTokens, generated.size(), nPromptTokens + generated.size());

        std::string chatJson = ((json)completion).dump();
        
        std::ostringstream oss;

        oss << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Content-Length: " << chatJson.length() << "\r\n\r\n" << chatJson;

        std::string response = oss.str();

        client_socket.write(response.c_str(), response.length());
    } else {
        outputChatCompletionChunk(client_socket, "", "stop");
    }
}

void server(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, AppArgs* args, TransformerSpec* spec) {
    SocketServer* server = new SocketServer(args->port);

    std::vector<Route> routes = {
        {
            "/v1/chat/completions",
            HttpMethod::METHOD_POST,
            std::bind(&handleCompletionsRequest, std::placeholders::_1, std::placeholders::_2, inference, tokenizer, sampler, args, spec)
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
            printf("New Request: %s %s\n", request.getMethod().c_str(), request.path.c_str());
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
