#ifndef API_TYPES_HPP
#define API_TYPES_HPP

#include <string>

#include "json.hpp"

using json = nlohmann::json;

struct ChatMessageDelta {
    std::string role;
    std::string content;

    ChatMessageDelta() : role(""), content("") {}
    ChatMessageDelta(const std::string& role_, const std::string& content_) : role(role_), content(content_) {}
};

struct ChatMessage {
    std::string role;
    std::string content;

    ChatMessage() : role(""), content("") {}
    ChatMessage(const std::string& role_, const std::string& content_) : role(role_), content(content_) {}
};

struct ChunkChoice {
    int index;
    ChatMessageDelta delta;
    std::string finish_reason;

    ChunkChoice() : index(0) {}
};


struct Choice {
    int index;
    ChatMessage message;
    std::string finish_reason;

    Choice() : finish_reason("") {}
    Choice(ChatMessage &message_) : message(message_), finish_reason("") {}
    Choice(const std::string &reason_) : finish_reason(reason_) {}
};

struct ChatCompletionChunk {
    std::string id;
    std::string object;
    long long created;
    std::string model;
    std::vector<ChunkChoice> choices;

    ChatCompletionChunk(ChunkChoice &choice_) 
        : id("cmpl-c0"), object("chat.completion"), model("Distributed Model") {
        created = std::time(nullptr); // Set created to current Unix timestamp
        choices.push_back(choice_);
    }
};

// Struct to represent the usage object
struct ChatUsage {
    int prompt_tokens;
    int completion_tokens;
    int total_tokens;

    ChatUsage() : prompt_tokens(0), completion_tokens(0), total_tokens(0) {}
    ChatUsage(int pt, int ct, int tt) : prompt_tokens(pt), completion_tokens(ct), total_tokens(tt) {}
};

struct ChatCompletion {
    std::string id;
    std::string object;
    long long created; // Unix timestamp
    std::string model;
    std::vector<Choice> choices;
    ChatUsage usage;

    ChatCompletion() : id(), object(), model() {}
    ChatCompletion(const Choice &choice_, const ChatUsage& usage_) 
        : id("cmpl-j0"), object("chat.completion"), model("Distributed Model"), usage(usage_) {
        created = std::time(nullptr); // Set created to current Unix timestamp
        choices.push_back(choice_);
    }
};

struct Model {
    std::string id;
    std::string object;
    long long created;
    std::string owned_by;

    Model() : id(), object(), created(0), owned_by() {}
    Model(const std::string &id_) : id(id_), object("model"), created(0), owned_by("user") {}
};

struct ModelList {
    std::string object;
    std::vector<Model> data;
    ModelList(): object("list") {}
    ModelList(const Model &model_) : object("list") {
        data.push_back(model_);
    }
};

struct InferenceParams {
    std::vector<ChatMessage> messages;
    int max_tokens;
    float temperature;
    float top_p;
    std::vector<std::string> stop;
    bool stream;
    unsigned long long seed;
};

// Define to_json for Delta struct
void to_json(json& j, const ChatMessageDelta& msg) {
    j = json{{"role", msg.role}, {"content", msg.content}};
}

void to_json(json& j, const ChatMessage& msg) {
    j = json{{"role", msg.role}, {"content", msg.content}};
}

void to_json(json& j, const ChunkChoice& choice) {
    j = json{{"index", choice.index}, {"delta", choice.delta}, {"finish_reason", choice.finish_reason}};
}

void to_json(json& j, const Choice& choice) {
    j = json{{"index", choice.index}, {"message", choice.message}, {"finish_reason", choice.finish_reason}};
}

void to_json(json& j, const ChatCompletionChunk& completion) {
    j = json{{"id", completion.id},
        {"object", completion.object},
        {"created", completion.created},
        {"model", completion.model},
        {"choices", completion.choices}};
}

void to_json(json& j, const ChatUsage& usage) {
    j = json{{"completion_tokens", usage.completion_tokens},
        {"prompt_tokens", usage.prompt_tokens},
        {"total_tokens", usage.total_tokens}};
}

void to_json(json& j, const ChatCompletion& completion) {
    j = json{{"id", completion.id},
        {"object", completion.object},
        {"created", completion.created},
        {"model", completion.model},
        {"usage", completion.usage},
        {"choices", completion.choices}};
}

void to_json(json& j, const Model& model) {
    j = json{{"id", model.id},
        {"object", model.object},
        {"created", model.created},
        {"owned_by", model.owned_by}};
}

void to_json(json& j, const ModelList& models) {
    j = json{{"object", models.object},
        {"data", models.data}};
}

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

#endif
