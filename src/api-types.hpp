#ifndef API_TYPES_HPP
#define API_TYPES_HPP

#include <string>

#include "json.hpp"

using json = nlohmann::json;

struct ToolFunctionDef {
    std::string name;
    std::string description;
    json parameters;

    ToolFunctionDef() : name(""), description(""), parameters(json::object()) {}
};

struct Tool {
    std::string type;
    ToolFunctionDef function;

    Tool() : type("function"), function() {}
};

struct ToolCallFunction {
    std::string name;
    std::string arguments;

    ToolCallFunction() : name(""), arguments("") {}
};

struct ToolCall {
    std::string id;
    std::string type;
    ToolCallFunction function;

    ToolCall() : id(""), type("function"), function() {}
};

struct ToolCallDelta {
    std::string id;
    std::string type;
    ToolCallFunction function;
    bool has_function;

    ToolCallDelta() : id(""), type("function"), function(), has_function(false) {}
};

struct ChatMessageDelta {
    std::string role;
    std::string content;
    std::vector<ToolCallDelta> tool_calls;
    bool has_tool_calls;

    ChatMessageDelta() : role(""), content(""), has_tool_calls(false) {}
    ChatMessageDelta(const std::string& role_, const std::string& content_) : role(role_), content(content_), has_tool_calls(false) {}
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;

    ChatMessage() : role(""), content(""), tool_call_id("") {}
    ChatMessage(const std::string& role_, const std::string& content_) : role(role_), content(content_), tool_call_id("") {}
};

struct ChunkChoice {
    int index;
    ChatMessageDelta delta;
    bool has_delta;
    std::string finish_reason;

    ChunkChoice() : index(0), delta(), has_delta(false) {}
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

enum ToolChoiceKind {
    TOOL_CHOICE_AUTO = 0,
    TOOL_CHOICE_NONE = 1,
    TOOL_CHOICE_REQUIRED = 2,
    TOOL_CHOICE_NAMED = 3
};

struct ToolChoice {
    ToolChoiceKind kind;
    std::string tool_name;

    ToolChoice() : kind(TOOL_CHOICE_NONE), tool_name("") {}
};

struct InferenceParams {
    std::vector<ChatMessage> messages;
    int max_tokens;
    float temperature;
    float top_p;
    std::vector<std::string> stop;
    bool stream;
    unsigned long long seed;
    std::vector<Tool> tools;
    ToolChoice tool_choice;
};

void to_json(json& j, const ToolFunctionDef& fn) {
    j = json{{"name", fn.name}, {"description", fn.description}, {"parameters", fn.parameters}};
}

void to_json(json& j, const Tool& tool) {
    j = json{{"type", tool.type}, {"function", tool.function}};
}

void to_json(json& j, const ToolCallFunction& fn) {
    j = json{{"name", fn.name}, {"arguments", fn.arguments}};
}

void to_json(json& j, const ToolCall& call) {
    j = json{{"id", call.id}, {"type", call.type}, {"function", call.function}};
}

void to_json(json& j, const ToolCallDelta& call) {
    j = json{{"id", call.id}, {"type", call.type}};
    if (call.has_function)
        j["function"] = call.function;
}

// Define to_json for Delta struct
void to_json(json& j, const ChatMessageDelta& msg) {
    j = json{{"role", msg.role}};
    if (!msg.content.empty())
        j["content"] = msg.content;
    if (msg.has_tool_calls && !msg.tool_calls.empty()) {
        json calls = json::array();
        for (const auto &call : msg.tool_calls)
            calls.push_back(call);
        j["tool_calls"] = calls;
    }
}

void to_json(json& j, const ChatMessage& msg) {
    j = json{{"role", msg.role}, {"content", msg.content}};
    if (!msg.tool_call_id.empty())
        j["tool_call_id"] = msg.tool_call_id;
    if (!msg.tool_calls.empty()) {
        json calls = json::array();
        for (const auto &call : msg.tool_calls)
            calls.push_back(call);
        j["tool_calls"] = calls;
    }
}

void to_json(json& j, const ChunkChoice& choice) {
    j = json{{"index", choice.index}, {"finish_reason", choice.finish_reason}};
    if (choice.has_delta) {
        j["delta"] = choice.delta;
    }
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
        ChatMessage msg;
        msg.role = item["role"].template get<std::string>();
        if (item.contains("content") && !item["content"].is_null())
            msg.content = item["content"].template get<std::string>();
        if (item.contains("tool_call_id"))
            msg.tool_call_id = item["tool_call_id"].template get<std::string>();
        if (item.contains("tool_calls") && item["tool_calls"].is_array()) {
            for (const auto& callItem : item["tool_calls"]) {
                ToolCall call;
                if (callItem.contains("id"))
                    call.id = callItem["id"].template get<std::string>();
                if (callItem.contains("type"))
                    call.type = callItem["type"].template get<std::string>();
                if (callItem.contains("function")) {
                    const auto& fn = callItem["function"];
                    if (fn.contains("name"))
                        call.function.name = fn["name"].template get<std::string>();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string())
                            call.function.arguments = fn["arguments"].template get<std::string>();
                        else
                            call.function.arguments = fn["arguments"].dump();
                    }
                }
                msg.tool_calls.push_back(call);
            }
        }
        messages.emplace_back(msg);
    }
    return messages;
}

InferenceParams parseInferenceParams(json &json, float defaultTemperature, float defaultTopp, unsigned long long defaultSeed) {
    InferenceParams params;
    params.temperature = defaultTemperature;
    params.top_p = defaultTopp;
    params.seed = defaultSeed;
    params.stream = false;
    params.messages = parseChatMessages(json["messages"]);
    params.max_tokens = -1;
    params.tools.clear();
    params.tool_choice = ToolChoice();

    if (json.contains("stream"))
        params.stream = json["stream"].get<bool>();
    if (json.contains("temperature"))
        params.temperature = json["temperature"].template get<float>();
    if (json.contains("seed"))
        params.seed = json["seed"].template get<unsigned long long>();
    if (json.contains("max_tokens"))
        params.max_tokens = json["max_tokens"].template get<int>();
    if (json.contains("tools") && json["tools"].is_array()) {
        for (const auto &toolItem : json["tools"]) {
            Tool tool;
            if (toolItem.contains("type"))
                tool.type = toolItem["type"].template get<std::string>();
            if (toolItem.contains("function")) {
                const auto &fn = toolItem["function"];
                if (fn.contains("name"))
                    tool.function.name = fn["name"].template get<std::string>();
                if (fn.contains("description"))
                    tool.function.description = fn["description"].template get<std::string>();
                if (fn.contains("parameters"))
                    tool.function.parameters = fn["parameters"];
            }
            params.tools.push_back(tool);
        }
        if (!params.tools.empty())
            params.tool_choice.kind = TOOL_CHOICE_AUTO;
    }
    if (json.contains("tool_choice")) {
        const auto &choice = json["tool_choice"];
        if (choice.is_string()) {
            std::string mode = choice.template get<std::string>();
            if (mode == "none") {
                params.tool_choice.kind = TOOL_CHOICE_NONE;
            } else if (mode == "required") {
                params.tool_choice.kind = TOOL_CHOICE_REQUIRED;
            } else {
                params.tool_choice.kind = TOOL_CHOICE_AUTO;
            }
        } else if (choice.is_object()) {
            if (choice.contains("type") && choice["type"] == "function" && choice.contains("function")) {
                const auto &fn = choice["function"];
                if (fn.contains("name")) {
                    params.tool_choice.kind = TOOL_CHOICE_NAMED;
                    params.tool_choice.tool_name = fn["name"].template get<std::string>();
                }
            }
        }
    }
    if (json.contains("stop")) {
        params.stop = json["stop"].template get<std::vector<std::string>>();
    } else {
        const std::string defaultStop = "<|eot_id|>";
        params.stop = std::vector<std::string>{defaultStop};
    }
    return params;
}

std::string normalizeToolArguments(const json &args) {
    if (args.is_string())
        return args.get<std::string>();
    return args.dump();
}

ToolCall parseToolCallJson(const json &callItem, size_t index) {
    ToolCall call;
    if (callItem.contains("id"))
        call.id = callItem["id"].template get<std::string>();
    if (callItem.contains("type"))
        call.type = callItem["type"].template get<std::string>();
    if (callItem.contains("function")) {
        const auto &fn = callItem["function"];
        if (fn.contains("name"))
            call.function.name = fn["name"].template get<std::string>();
        if (fn.contains("arguments"))
            call.function.arguments = normalizeToolArguments(fn["arguments"]);
    } else {
        if (callItem.contains("name"))
            call.function.name = callItem["name"].template get<std::string>();
        if (callItem.contains("arguments"))
            call.function.arguments = normalizeToolArguments(callItem["arguments"]);
    }
    if (call.type.empty())
        call.type = "function";
    if (call.id.empty())
        call.id = "call_" + std::to_string(index + 1);
    return call;
}

bool isValidToolCallJson(const json &callItem) {
    if (!callItem.is_object())
        return false;
    if (callItem.contains("function")) {
        const auto &fn = callItem["function"];
        if (!fn.is_object())
            return false;
        if (!fn.contains("name") || !fn["name"].is_string())
            return false;
        return true;
    }
    if (callItem.contains("name") && callItem["name"].is_string())
        return true;
    return false;
}

bool tryParseToolCallsFromJson(const json &data, std::vector<ToolCall> &toolCalls) {
    if (data.is_object() && data.contains("tool_calls")) {
        const auto &calls = data["tool_calls"];
        if (!calls.is_array())
            return false;
        bool any = false;
        for (size_t i = 0; i < calls.size(); i++) {
            if (!isValidToolCallJson(calls[i]))
                continue;
            toolCalls.push_back(parseToolCallJson(calls[i], i));
            any = true;
        }
        return any;
    }
    if (data.is_object() && data.contains("function_call")) {
        const auto &call = data["function_call"];
        if (!isValidToolCallJson(call))
            return false;
        toolCalls.push_back(parseToolCallJson(call, 0));
        return true;
    }
    return false;
}

#endif
