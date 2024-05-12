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

#include "../utils.hpp"
#include "../socket.hpp"
#include "../transformer.hpp"
#include "../tasks.hpp"
#include "../llama2-tasks.hpp"
#include "../grok1-tasks.hpp"
#include "../mixtral-tasks.hpp"
#include "../tokenizer.hpp"
#include "../http.hpp"
#include "../common/json.hpp"

constexpr int BUFFER_SIZE = 8192;
constexpr float DEFAULT_TEMPERATURE = 0.8f;
constexpr float DEFAULT_TOPP = 0.9f;

struct ServerArgs {
    int nThreads; 

    // inference
    char* modelPath;
    char* tokenizerPath;
    FloatType weightsFloatType;
    FloatType bufferFloatType;
    int nWorkers;
    char** workerHosts;
    int* workerPorts;

    // server
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

FloatType parseFloatType(char* val) {
    if (strcmp(val, "f32") == 0) return F32;
    if (strcmp(val, "f16") == 0) return F16;
    if (strcmp(val, "q40") == 0) return Q40;
    if (strcmp(val, "q80") == 0) return Q80;
    printf("Invalid float type %s\n", val);
    exit(EXIT_FAILURE);
}

struct ChatMessageDelta {
    std::string role;
    std::string content;
};

// Define to_json for Delta struct
void to_json(json& j, const ChatMessageDelta& msg) {
    j = json{{"role", msg.role}, {"content", msg.content}};
}

struct ChatMessage {
    std::string role;
    std::string content;
};

// Define to_json for ChatMessage struct
void to_json(json& j, const ChatMessage& msg) {
    j = json{{"role", msg.role}, {"content", msg.content}};
}

struct ChunkChoice {
    int index;
    ChatMessageDelta delta;
    std::string finish_reason;
};

// Define to_json for ChunkChoice struct
void to_json(json& j, const ChunkChoice& choice) {
    j = json{{"index", choice.index}, {"delta", choice.delta}, {"finish_reason", choice.finish_reason}};
}

struct Choice {
    int index;
    ChatMessage message;
    std::string finish_reason;
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
};

// Struct to represent the chat completion object
struct ChatCompletion {
    std::string id;
    std::string object;
    long long created; // Assuming Unix timestamp
    std::string model;
    std::vector<Choice> choices;
    ChatUsage usage;
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

    for (const auto& item : json) {
        ChatMessage message;
        message.content = item["content"].template get<std::string>();
        message.role = item["role"].template get<std::string>();
        messages.push_back(message);
    }

    return messages;
}

/*
Generally speaking, the tokenizer.config.json would contain the chat template for the model
and depending on the model used you could set the chat template to follow
could possibly just for simplicity set this in ServerArgs with --chat-template
for this code draft I am assuming the use of llama 3 instruct
*/
std::string buildChatPrompt(Tokenizer *tokenizer, std::vector<ChatMessage> &messages){
    std::ostringstream oss;

    //oss << tokenizer->decode(-1, tokenizer->bosId);

    for (const auto& message : messages) {
        oss << "<|start_header_id|>" << message.role << "<|end_header_id|>\n\n" << message.content << "<|eot_id|>";
    }

    oss << "<|start_header_id|>assistant<|end_header_id|>\n\n";

    return oss.str();
}

void outputChatCompletionChunk(Socket &client_socket, std::string delta, std::string finish_reason = ""){
    ChunkChoice choice;
    choice.index = 0;
    
    if(finish_reason.size() > 0){
        choice.finish_reason = finish_reason;
    }
    else{
        ChatMessageDelta responseDelta;
        responseDelta.role = "assistant";
        responseDelta.content = delta;
        choice.delta = responseDelta;
    }
    
    std::vector<ChunkChoice> choices;
    choices.push_back(choice);

    ChatCompletionChunk chunk;
    chunk.id = "chatcmpl-test";
    chunk.object = "chat.completion";
    chunk.model = "Distributed Model";
    chunk.created = time_t();
    chunk.choices = choices;
    
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
    //send(client_socket, formattedChunk.str().c_str(), formattedChunk.str().length(), 0);
}

void completeChat(Socket &client_socket, InferenceParams &request, Inference* inference, Tokenizer *tokenizer, Sampler *sampler, TransformerSpec* spec) {
    std::vector<std::string> generated;
    generated.get_allocator().allocate(request.max_tokens);

    if (request.stream) {
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: text/event-stream; charset=utf-8\r\n"
            << "Connection: keep-alive\r\n"
            << "Transfer-Encoding: chunked\r\n\r\n";
            
        client_socket.write(oss.str().c_str(), oss.str().length());
        //send(client_socket, oss.str().c_str(), oss.str().length(), 0);
    }

    int promptLength = request.prompt.length();
    int nPromptTokens;
    int promptTokens[promptLength + 3];
    char prompt[promptLength + 1];
    prompt[promptLength] = 0;
    strcpy(prompt, request.prompt.c_str());
    tokenizer->encode(prompt, promptTokens, &nPromptTokens, true, false);

    int token = promptTokens[0];
    pos_t maxPos = nPromptTokens + request.max_tokens;
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
            
            if (!request.stop.empty() && safePiece) {
                std::string concatenatedTokens;
                int startIndex = std::max(0, static_cast<int>(generated.size()) - 7);
                for (int i = startIndex; i < generated.size(); ++i) {
                    concatenatedTokens += generated[i];
                }
                concatenatedTokens += std::string(piece);

                for (const auto& word : request.stop) {
                    if (concatenatedTokens.find(word) != std::string::npos) {
                        eosEncountered = true;
                        break;
                    }
                }
            }

            if (eosEncountered) break;

            char string[100];
            strcpy(string, piece);

            generated.push_back(std::string(string));
            
            if (request.stream) {
                outputChatCompletionChunk(client_socket, std::string(string));
            }
        }
    }

    if (!request.stream) {
        std::vector<Choice> choices;
        ChatMessage responseMessage;
        responseMessage.role = "assistant";
        responseMessage.content = std::accumulate(generated.begin(), generated.end(), std::string(""));
        Choice responseChoice;
        responseChoice.message = responseMessage;
        choices.push_back(responseChoice);
        ChatCompletion completion;
        completion.id = "chatcmpl-test";
        completion.object = "chat.completion";
        completion.model = "Distributed Model";
        completion.created = time_t();
        completion.choices = choices;
        ChatUsage usage;
        usage.prompt_tokens = nPromptTokens;
        usage.completion_tokens = generated.size();
        usage.total_tokens = nPromptTokens + generated.size();
        completion.usage = usage;

        std::string response = ((json)completion).dump();
        
        std::ostringstream oss;
        oss << "HTTP/1.1 200 OK\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Content-Length: " << response.length() << "\r\n\r\n";

        std::string header = oss.str();
        response = header + response;

        client_socket.write(response.c_str(), response.length());
        //send(client_socket, response.c_str(), response.length(), 0);
    }
    else{
        outputChatCompletionChunk(client_socket, "", "stop");
    }
}

void handleClient(Socket &client_socket, Inference* inference, Tokenizer *tokenizer, Sampler *sampler, ServerArgs* args, TransformerSpec* spec){
    char buffer[BUFFER_SIZE] = {0};

    client_socket.read((char*)&buffer, BUFFER_SIZE);

    HTTP::HttpRequest request = HTTP::HttpParser::parseRequest(std::string(buffer));

    printf("New Request: %s %s\n", request.getMethod().c_str(), request.path.c_str());

    if(request.method == HTTP::HttpMethod::METHOD_POST && request.path == "/v1/chat/completions"){
        InferenceParams inferParams;
        inferParams.stream = false;
        inferParams.max_tokens = 8192;
        inferParams.top_p = DEFAULT_TOPP;
        inferParams.temperature = DEFAULT_TEMPERATURE;
        
        std::vector<ChatMessage> messages = parseChatMessages(request.parsedJson["messages"]);
        inferParams.prompt = buildChatPrompt(tokenizer, messages);
        
        if(request.parsedJson.contains("stream")){
            inferParams.stream = request.parsedJson["stream"].template get<bool>();
        }
        if(request.parsedJson.contains("temperature")){
            inferParams.temperature = request.parsedJson["temperature"].template get<float>();
            sampler->setTemp(inferParams.temperature);
        }
        if(request.parsedJson.contains("seed")){
            inferParams.seed = request.parsedJson["seed"].template get<unsigned long long>();
            sampler->setSeed(inferParams.seed);
        }
        else{
            sampler->setSeed((unsigned long long)time(NULL));
        }
        if(request.parsedJson.contains("max_tokens")){
            inferParams.max_tokens = request.parsedJson["max_tokens"].template get<int>();
        }
        if(request.parsedJson.contains("stop")){
            inferParams.stop = request.parsedJson["stop"].template get<std::vector<std::string>>();
        }

        completeChat(client_socket, inferParams, inference, tokenizer, sampler, spec);
    }
    else{
        std::string header = "HTTP/1.1 404 Not Found\r\n";
        client_socket.write(header.c_str(), header.length());
        //send(client_socket, header.c_str(), header.length(), 0);
    }
}

void server(Inference* inference, SocketPool* socketPool, Tokenizer *tokenizer, Sampler *sampler, ServerArgs* args, TransformerSpec* spec) {
    SocketServer* server = new SocketServer(args->port);

    while (true) {
        try {
            // Accept incoming connection
            Socket client = server->accept();

            handleClient(client, inference, tokenizer, sampler, args, spec);

            // Close client socket
            delete &client;
        } catch (ReadSocketException& ex) {
            printf("Read socket error: %d %s\n", ex.code, ex.message);
        } catch (WriteSocketException& ex) {
            printf("Write socket error: %d %s\n", ex.code, ex.message);
        }
    }

    delete server;
}

int run(ServerArgs* args, void (*program)(Inference* inference, SocketPool* socketPool, Tokenizer* tokenizer, Sampler* sampler, ServerArgs* args, TransformerSpec* spec)) {
    if (args->modelPath == NULL) {
        return usage("Model is required");
    }
    if (args->tokenizerPath == NULL) {
        return usage("Tokenizer is required");
    }

    int defaultSeed = (unsigned long long)time(NULL);

    SocketPool* socketPool = SocketPool::connect(args->nWorkers, args->workerHosts, args->workerPorts);
    unsigned int nSlices = args->nWorkers + 1;

    TransformerSpec spec = Transformer::loadSpecFromFile(args->modelPath, nSlices, args->weightsFloatType, args->bufferFloatType);
    TransformerArch arch = getArch(&spec);

    Tokenizer tokenizer(args->tokenizerPath, spec.vocabSize);
    Transformer transformer = Transformer::loadRootFromFile(args->modelPath, &spec, socketPool);
    Inference inference = Inference(&arch, args->nThreads, &transformer, socketPool);

    Sampler sampler(spec.vocabSize, DEFAULT_TEMPERATURE, DEFAULT_TOPP, defaultSeed);

    program(&inference, socketPool, &tokenizer, &sampler, args, &spec);

    delete socketPool;
    return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
    initQuants();
    
    ServerArgs args;
    
    args.nThreads = 4;
    args.modelPath = NULL;
    args.tokenizerPath = NULL;
    args.weightsFloatType = F32;
    args.bufferFloatType = F32;
    args.nWorkers = 0;
    args.port = 8080;
    
    for (int i = 1; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--model") == 0) {
            args.modelPath = argv[i + 1];
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            args.tokenizerPath = argv[i + 1];
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
        } else {
            printf("Unknown option %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }

    return run(&args, server);
}