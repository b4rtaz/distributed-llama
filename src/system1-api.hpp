#ifndef SYSTEM1_API_HPP
#define SYSTEM1_API_HPP

#include <string>
#include <vector>

#include "json.hpp"
#include "nn/nn-core.hpp"

using System1Json = nlohmann::ordered_json;

struct System1Question {
    std::string id;
    std::string type;
    System1Json instructions;
    System1Json criteria;
    bool hasAllowNone;
    bool allowNone;

    System1Question() : hasAllowNone(false), allowNone(false) {}
};

struct System1Request {
    System1Json state;
    std::vector<System1Question> questions;
    bool allowNone;
    bool noneCheck;
    bool returnDetails;

    System1Request()
        : state(nullptr), allowNone(false), noneCheck(true), returnDetails(false) {}
};

struct System1Usage {
    NnUint inputTokens;
    NnUint outputTokens;

    System1Usage() : inputTokens(0), outputTokens(0) {}
};

struct System1Response {
    System1Json answers;
    System1Usage usage;
    std::string model;
    System1Json timingMs;
    bool hasTiming;

    System1Response() : answers(System1Json::object()), timingMs(System1Json::object()), hasTiming(false) {}
};

inline void to_json(System1Json &j, const System1Usage &usage) {
    j = System1Json{
        {"input_tokens", usage.inputTokens},
        {"output_tokens", usage.outputTokens},
    };
}

inline void to_json(System1Json &j, const System1Response &response) {
    j = System1Json{
        {"answers", response.answers},
        {"usage", response.usage},
        {"model", response.model},
    };
    if (response.hasTiming)
        j["timing_ms"] = response.timingMs;
}

#endif
