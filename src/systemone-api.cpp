#include <algorithm>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "app.hpp"
#include "graphs/qwen35-graph.hpp"
#include "http.hpp"
#include "nn/nn-network.hpp"
#include "systemone-api.hpp"
#include "tokenizer.hpp"

static const char *SYSTEM_PROMPT =
    "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only.";
static const char *NONE_OPTION = "None of the above";
static const NnUint N_LETTERS = 26u;

class System1RequestException : public std::runtime_error {
public:
    std::string field;

    System1RequestException(const std::string &message, const std::string &field)
        : std::runtime_error(message), field(field) {}
};

struct System1DecisionConfig {
    std::vector<NnUint> letterTokenIds;
    float choiceTemperature;
    float noulTemperature;
    float scoreTemperature;
    NnUint maxOptions;
    NnUint wideGroup;
    NnUint wideKeep;
    float wideResidual;
    std::string releaseDate;

    System1DecisionConfig()
        : choiceTemperature(1.0f), noulTemperature(1.0f), scoreTemperature(1.0f),
          maxOptions(26u), wideGroup(25u), wideKeep(3u), wideResidual(0.001f) {}
};

struct DecisionOption {
    std::string id;
    std::string criterion;
    bool hasCriterion;

    DecisionOption(const std::string &id_, const std::string &criterion_, bool hasCriterion_)
        : id(id_), criterion(criterion_), hasCriterion(hasCriterion_) {}
};

struct DecisionRow {
    std::string type;
    std::string state;
    std::string instructions;
    std::vector<DecisionOption> options;
    bool hasNoneOption;

    DecisionRow() : hasNoneOption(false) {}
};

struct Readout {
    std::vector<float> probabilities;
    NnUint inputTokens;
    double milliseconds;

    Readout() : inputTokens(0u), milliseconds(0.0) {}
};

static System1DecisionConfig decisionConfig;

static std::string jsonValueText(const System1Json &value, bool *hasValue = nullptr) {
    if (value.is_null()) {
        if (hasValue != nullptr) *hasValue = false;
        return "";
    }
    std::string text = value.is_string() ? value.get<std::string>() : value.dump();
    const size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        if (hasValue != nullptr) *hasValue = false;
        return "";
    }
    const size_t end = text.find_last_not_of(" \t\r\n");
    if (hasValue != nullptr) *hasValue = true;
    return text.substr(begin, end - begin + 1u);
}

static System1Json annotateArrayIndexes(const System1Json &value) {
    if (value.is_array()) {
        System1Json result = System1Json::array();
        const bool annotate = value.size() >= 8u;
        for (size_t i = 0; i < value.size(); i++) {
            System1Json item = annotateArrayIndexes(value[i]);
            if (!annotate) {
                result.push_back(item);
            } else if (item.is_object()) {
                System1Json indexed = System1Json::object();
                indexed["_index"] = i;
                for (System1Json::const_iterator it = item.begin(); it != item.end(); ++it)
                    indexed[it.key()] = it.value();
                result.push_back(indexed);
            } else {
                result.push_back(System1Json{{"_index", i}, {"value", item}});
            }
        }
        return result;
    }
    if (value.is_object()) {
        System1Json result = System1Json::object();
        for (System1Json::const_iterator it = value.begin(); it != value.end(); ++it)
            result[it.key()] = annotateArrayIndexes(it.value());
        return result;
    }
    return value;
}

static std::string stateText(const System1Json &state) {
    if (state.is_string())
        return state.get<std::string>();
    if (state.is_null() || (state.is_object() && state.empty()) || (state.is_array() && state.empty()))
        return "";
    return annotateArrayIndexes(state).dump();
}

static std::string normalizedText(const std::string &value) {
    std::string result;
    bool wasSpace = false;
    for (size_t i = 0; i < value.size(); i++) {
        const unsigned char c = (unsigned char)value[i];
        const bool space = std::isspace(c) || c == '_' || c == '-';
        if (space) {
            if (!result.empty() && !wasSpace)
                result.push_back(' ');
        } else {
            result.push_back((char)std::tolower(c));
        }
        wasSpace = space;
    }
    if (!result.empty() && result[result.size() - 1u] == ' ')
        result.resize(result.size() - 1u);
    return result;
}

static bool isBareOptionId(const std::string &id) {
    static const std::regex bare(
        "^(\\(?[A-Za-z][\\).]?|\\(?[0-9]{1,3}[\\).]?|option[_ ]?[0-9]{1,3}|opt[_ ]?[0-9]{1,3})$",
        std::regex_constants::icase);
    return std::regex_match(id, bare);
}

static std::vector<std::string> optionLines(const DecisionRow &row) {
    bool hideIds = row.type == "choice";
    const size_t listedOptions = row.hasNoneOption ? row.options.size() - 1u : row.options.size();
    for (size_t i = 0; i < listedOptions; i++) {
        if (!isBareOptionId(row.options[i].id) || !row.options[i].hasCriterion) {
            hideIds = false;
            break;
        }
    }

    std::vector<std::string> lines;
    for (size_t i = 0; i < row.options.size(); i++) {
        const DecisionOption &option = row.options[i];
        std::string name = option.id;
        if (row.type == "noul")
            name = option.id == "true" ? "yes" : "no";

        std::string body;
        if (hideIds && i < listedOptions) {
            body = option.criterion;
        } else if (!option.hasCriterion || normalizedText(option.criterion) == normalizedText(name)) {
            body = name;
        } else {
            body = name + ": " + option.criterion;
        }
        lines.push_back(std::string(1u, (char)('A' + i)) + ") " + body);
    }
    return lines;
}

static std::string renderPrompt(const DecisionRow &row) {
    const std::vector<std::string> lines = optionLines(row);
    std::ostringstream user;
    user << "Evidence:\n" << (row.state.empty() ? "(none)" : row.state)
         << "\n\nQuestion: " << row.instructions << "\nOptions:\n";
    for (size_t i = 0; i < lines.size(); i++) {
        if (i > 0u) user << "\n";
        user << lines[i];
    }

    return std::string("<|im_start|>system\n") + SYSTEM_PROMPT + "<|im_end|>\n" +
        "<|im_start|>user\n" + user.str() + "<|im_end|>\n" +
        "<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

static bool jsonBool(const System1Json &object, const char *key, bool fallback) {
    if (!object.contains(key) || object[key].is_null())
        return fallback;
    if (!object[key].is_boolean())
        throw System1RequestException(std::string(key) + " must be true or false", key);
    return object[key].get<bool>();
}

static void validateNumberControl(const System1Json &root, const char *key) {
    if (!root.contains(key) || root[key].is_null())
        return;
    if (!root[key].is_number())
        throw System1RequestException(std::string(key) + " must be a number from 0 to 1", key);
    const double value = root[key].get<double>();
    if (!std::isfinite(value) || value < 0.0 || value > 1.0)
        throw System1RequestException(std::string(key) + " must be a number from 0 to 1", key);
}

static System1Request parseSystem1Request(const std::string &body) {
    System1Json root = System1Json::parse(body, nullptr, false);
    if (root.is_discarded() || !root.is_object())
        throw System1RequestException("the request body must be a JSON object", "body");
    if (root.contains("images") && (!root["images"].is_array() || !root["images"].empty()))
        throw System1RequestException("images are not supported by this inference engine", "images");
    if (root.contains("think_mode") && !root["think_mode"].is_null()) {
        if (!root["think_mode"].is_string())
            throw System1RequestException("think_mode must be a string", "think_mode");
        const std::string mode = root["think_mode"].get<std::string>();
        if (mode != "never" && mode != "plain")
            throw System1RequestException(
                "this is an Instant model: it does not think; use think_mode \"never\" or \"plain\"",
                "think_mode");
    }
    if (jsonBool(root, "return_thought", false))
        throw System1RequestException("Instant models do not produce thoughts", "return_thought");
    validateNumberControl(root, "gate");
    validateNumberControl(root, "mix");
    if (root.contains("think_budget") && !root["think_budget"].is_null() &&
        (!root["think_budget"].is_number_integer() || root["think_budget"].get<long long>() < 1ll))
        throw System1RequestException("think_budget must be a positive integer", "think_budget");
    if (root.contains("thought_generation") && !root["thought_generation"].is_null()) {
        if (!root["thought_generation"].is_string())
            throw System1RequestException("thought_generation must be a string", "thought_generation");
        const std::string generation = root["thought_generation"].get<std::string>();
        if (generation != "auto" && generation != "stream" && generation != "chunked")
            throw System1RequestException(
                "thought_generation must be auto, stream, or chunked", "thought_generation");
    }
    if (!root.contains("questions") || !root["questions"].is_object() || root["questions"].empty())
        throw System1RequestException("questions must be a non-empty object", "questions");

    System1Request request;
    request.state = root.contains("state") ? root["state"] : System1Json(nullptr);
    request.allowNone = jsonBool(root, "allow_none", false);
    request.noneCheck = jsonBool(root, "none_check", true);
    request.returnDetails = jsonBool(root, "return_details", false);

    const System1Json &questions = root["questions"];
    for (System1Json::const_iterator it = questions.begin(); it != questions.end(); ++it) {
        if (!it.value().is_object())
            throw System1RequestException("each question must be an object", "questions." + it.key());
        const System1Json &spec = it.value();
        System1Question question;
        question.id = it.key();
        question.type = spec.value("type", std::string("choice"));
        if (question.type == "bool") question.type = "noul";
        question.instructions = spec.contains("instructions")
            ? spec["instructions"]
            : (spec.contains("question") ? spec["question"] : System1Json(nullptr));
        question.criteria = spec.contains("criteria")
            ? spec["criteria"]
            : (spec.contains("options") ? spec["options"] : System1Json(nullptr));
        if (spec.contains("allow_none") && !spec["allow_none"].is_null()) {
            if (!spec["allow_none"].is_boolean())
                throw System1RequestException("allow_none must be true or false", "questions." + it.key());
            question.hasAllowNone = true;
            question.allowNone = spec["allow_none"].get<bool>();
        }
        request.questions.push_back(question);
    }
    return request;
}

static bool parseNumber(const std::string &text, double *number) {
    char *end = nullptr;
    *number = std::strtod(text.c_str(), &end);
    return end != text.c_str() && *end == '\0' && std::isfinite(*number);
}

static std::vector<std::pair<std::string, System1Json>> orderedScoreLevels(const System1Json &criteria) {
    std::vector<std::pair<std::string, System1Json>> levels;
    for (System1Json::const_iterator it = criteria.begin(); it != criteria.end(); ++it)
        levels.push_back(std::make_pair(it.key(), it.value()));

    bool numeric = true;
    for (size_t i = 0; i < levels.size(); i++) {
        double unused;
        if (!parseNumber(levels[i].first, &unused)) numeric = false;
    }
    if (numeric) {
        std::stable_sort(levels.begin(), levels.end(),
            [](const std::pair<std::string, System1Json> &a,
               const std::pair<std::string, System1Json> &b) {
                double av, bv;
                parseNumber(a.first, &av);
                parseNumber(b.first, &bv);
                return av < bv;
            });
    }
    return levels;
}

static DecisionRow buildRow(
    const System1Json &state, const System1Question &question, bool addNone,
    const std::vector<DecisionOption> *choiceOptions = nullptr
) {
    DecisionRow row;
    row.type = question.type;
    row.state = stateText(state);
    row.instructions = jsonValueText(question.instructions);

    if (row.type == "choice") {
        if (choiceOptions != nullptr) {
            row.options = *choiceOptions;
        } else if (question.criteria.is_object()) {
            for (System1Json::const_iterator it = question.criteria.begin(); it != question.criteria.end(); ++it) {
                bool hasCriterion;
                const std::string criterion = jsonValueText(it.value(), &hasCriterion);
                row.options.push_back(DecisionOption(it.key(), criterion, hasCriterion));
            }
        } else if (question.criteria.is_array()) {
            for (size_t i = 0; i < question.criteria.size(); i++) {
                const std::string id = jsonValueText(question.criteria[i]);
                row.options.push_back(DecisionOption(id, "", false));
            }
        } else {
            throw System1RequestException(
                "a choice question needs criteria as an object or array", "questions." + question.id);
        }
        if (row.instructions.empty())
            throw System1RequestException(
                "instructions are required for choice questions", "questions." + question.id);
        if (row.options.empty())
            throw System1RequestException("a choice question needs at least one option", "questions." + question.id);

        std::set<std::string> ids;
        for (size_t i = 0; i < row.options.size(); i++) {
            if (row.options[i].id.empty() || row.options[i].id.find('\n') != std::string::npos ||
                row.options[i].id.find('\r') != std::string::npos || row.options[i].id.size() > 200u)
                throw System1RequestException("choice option ids must be non-empty single-line strings",
                    "questions." + question.id);
            if (!ids.insert(row.options[i].id).second)
                throw System1RequestException("choice option ids must be unique", "questions." + question.id);
        }
        if (addNone) {
            for (size_t i = 0; i < row.options.size(); i++) {
                if (normalizedText(row.options[i].id) == normalizedText(NONE_OPTION))
                    throw System1RequestException(
                        "the API adds None of the above automatically; remove that option",
                        "questions." + question.id);
            }
            row.options.push_back(DecisionOption(NONE_OPTION, "", false));
            row.hasNoneOption = true;
        }
    } else if (row.type == "noul") {
        if (row.instructions.empty())
            row.instructions = "Which answer fits the evidence?";
        System1Json criteria = question.criteria.is_object() ? question.criteria : System1Json::object();
        bool hasTrue, hasFalse;
        const std::string trueCriterion = criteria.contains("true")
            ? jsonValueText(criteria["true"], &hasTrue)
            : (hasTrue = false, std::string());
        const std::string falseCriterion = criteria.contains("false")
            ? jsonValueText(criteria["false"], &hasFalse)
            : (hasFalse = false, std::string());
        row.options.push_back(DecisionOption("true", trueCriterion, hasTrue));
        row.options.push_back(DecisionOption("false", falseCriterion, hasFalse));
    } else if (row.type == "score") {
        if (row.instructions.empty())
            throw System1RequestException(
                "instructions are required for score questions", "questions." + question.id);
        std::vector<System1Json> levels;
        if (question.criteria.is_array()) {
            for (size_t i = 0; i < question.criteria.size(); i++) levels.push_back(question.criteria[i]);
        } else if (question.criteria.is_object()) {
            const std::vector<std::pair<std::string, System1Json>> ordered =
                orderedScoreLevels(question.criteria);
            for (size_t i = 0; i < ordered.size(); i++) levels.push_back(ordered[i].second);
        } else {
            throw System1RequestException(
                "a score question needs criteria as an array or object", "questions." + question.id);
        }
        if (levels.size() < 2u || levels.size() > 10u)
            throw System1RequestException(
                "a score question needs between 2 and 10 levels", "questions." + question.id);
        for (size_t i = 0; i < levels.size(); i++) {
            bool hasCriterion;
            const std::string criterion = jsonValueText(levels[i], &hasCriterion);
            row.options.push_back(DecisionOption(std::to_string(i), criterion, hasCriterion));
        }
    } else {
        throw System1RequestException(
            "unknown question type; use choice, noul, bool, or score", "questions." + question.id);
    }
    return row;
}

static float temperatureForType(const std::string &type) {
    if (type == "choice") return decisionConfig.choiceTemperature;
    if (type == "noul") return decisionConfig.noulTemperature;
    return decisionConfig.scoreTemperature;
}

static std::vector<float> softmax(const std::vector<float> &logits, float temperature) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        probabilities[i] = std::exp((logits[i] - maximum) / temperature);
        sum += probabilities[i];
    }
    for (size_t i = 0; i < probabilities.size(); i++)
        probabilities[i] = (float)(probabilities[i] / sum);
    return probabilities;
}

static float choiceConfidence(const std::vector<float> &probabilities) {
    if (probabilities.size() < 2u) return 1.0f;
    const float maximum = *std::max_element(probabilities.begin(), probabilities.end());
    const float n = (float)probabilities.size();
    return std::max(0.0f, std::min(1.0f, (n * maximum - 1.0f) / (n - 1.0f)));
}

static float scoreConfidence(const std::vector<float> &probabilities) {
    const size_t n = probabilities.size();
    const size_t mode = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
    double spread = 0.0;
    double even = 0.0;
    const double middle = ((double)n - 1.0) / 2.0;
    for (size_t i = 0; i < n; i++) {
        spread += probabilities[i] * std::abs((double)i - (double)mode);
        even += std::abs((double)i - middle) / (double)n;
    }
    return even == 0.0 ? 1.0f : (float)std::max(0.0, 1.0 - spread / even);
}

static size_t maximumIndex(const std::vector<float> &values) {
    size_t index = 0u;
    for (size_t i = 1u; i < values.size(); i++) {
        if (values[i] > values[index]) index = i;
    }
    return index;
}

static double elapsedMilliseconds(const std::chrono::steady_clock::time_point &started) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
}

static System1Json keyedProbabilities(
    const DecisionRow &row, const std::vector<float> &probabilities
) {
    System1Json result = System1Json::object();
    for (size_t i = 0; i < probabilities.size(); i++)
        result[row.options[i].id] = probabilities[i];
    return result;
}

static System1Json instantDetails(
    const System1Json &probabilities, float top, double firstMilliseconds,
    double totalMilliseconds
) {
    return System1Json{
        {"first_pass", System1Json{
            {"probabilities", probabilities}, {"top", top}, {"ms", firstMilliseconds},
        }},
        {"thought", System1Json{
            {"used", false}, {"tokens", 0}, {"ended", "skipped"}, {"ms", 0.0},
        }},
        {"after_thought", nullptr},
        {"gate", nullptr},
        {"mix", nullptr},
        {"total_ms", totalMilliseconds},
    };
}

class System1Api {
private:
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    AppCliArgs *args;
    Qwen35LlmHeader *header;
    const char *modelName;

public:
    explicit System1Api(AppInferenceContext *context)
        : inference(context->inference), tokenizer(context->tokenizer), args(context->args),
          header(static_cast<Qwen35LlmHeader *>(context->header)),
          modelName(getFileName(context->args->modelPath)) {
        if (header->archType != QWEN35_STARTLUX_INSTANT)
            throw std::runtime_error("systemone-api requires a QWEN35_STARTLUX_INSTANT model");
        validateTokenizer();
    }

    void decide(HttpRequest &request) {
        const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        try {
            const System1Request parsed = parseSystem1Request(request.body);
            System1Response response;
            response.model = modelName;
            double firstPassMilliseconds = 0.0;
            for (size_t i = 0; i < parsed.questions.size(); i++) {
                const System1Question &question = parsed.questions[i];
                const bool allowNone = question.hasAllowNone ? question.allowNone : parsed.allowNone;
                NnUint questionTokens = 0u;
                double questionFirstMilliseconds = 0.0;
                response.answers[question.id] = answerQuestion(
                    parsed.state, question, allowNone, parsed.noneCheck, parsed.returnDetails,
                    &questionTokens, &questionFirstMilliseconds);
                response.usage.inputTokens += questionTokens;
                firstPassMilliseconds += questionFirstMilliseconds;
            }
            if (parsed.returnDetails) {
                const std::chrono::duration<double, std::milli> elapsed =
                    std::chrono::steady_clock::now() - started;
                response.hasTiming = true;
                response.timingMs["wall"] = elapsed.count();
                response.timingMs["first_pass_sum"] = firstPassMilliseconds;
                response.timingMs["thought_span"] = 0.0;
                response.timingMs["after_thought_sum"] = 0.0;
            }
            System1Json output;
            to_json(output, response);
            request.writeJson(output.dump());
        } catch (const System1RequestException &e) {
            writeError(request, 422, e.what(), e.field);
        } catch (const std::exception &e) {
            writeError(request, 500, e.what(), "server");
        }
    }

    void models(HttpRequest &request) const {
        System1Json model = {
            {"name", modelName},
            {"description", "StartLux Decision v1.1 9B Instant"},
            {"release_date", decisionConfig.releaseDate},
        };
        request.writeJson(System1Json{{"models", System1Json::array({model})}}.dump());
    }

    void health(HttpRequest &request) const {
        request.writeJson(System1Json{
            {"status", "ok"},
            {"model", modelName},
            {"kind", "instant"},
            {"prompt_attention", "bidirectional"},
            {"images", false},
            {"batch_capacity", args->nBatches},
            {"context_length", header->seqLen},
        }.dump());
    }

private:
    static void writeError(HttpRequest &request, int status, const std::string &message, const std::string &field) {
        System1Json location = System1Json::array();
        location.push_back("body");
        if (field != "body" && !field.empty()) location.push_back(field);
        request.writeJson(System1Json{
            {"error", message},
            {"detail", System1Json::array({System1Json{
                {"loc", location}, {"msg", message}, {"type", "value_error"},
            }})},
        }.dump(), status);
    }

    void validateTokenizer() const {
        if (decisionConfig.letterTokenIds.size() != N_LETTERS)
            throw std::runtime_error("decision_config.json must contain 26 letter token ids");
        std::set<NnUint> unique;
        for (NnUint i = 0; i < N_LETTERS; i++) {
            const NnUint token = decisionConfig.letterTokenIds[i];
            if (token >= tokenizer->vocabSize || !unique.insert(token).second)
                throw std::runtime_error("decision_config.json contains invalid letter token ids");
        }
    }

    Readout read(const DecisionRow &row) {
        if (row.options.size() < 2u || row.options.size() > decisionConfig.maxOptions)
            throw System1RequestException("a model pass needs between 2 and 26 options", "questions");
        const std::string prompt = renderPrompt(row);
        std::vector<int> tokens(prompt.size() + 2u);
        int nTokens = 0;
        tokenizer->encode((char *)prompt.c_str(), tokens.data(), &nTokens, true, true);
        if (nTokens <= 0)
            throw System1RequestException("the rendered prompt is empty", "questions");
        if ((NnUint)nTokens > args->nBatches) {
            throw System1RequestException(
                "the prompt has " + std::to_string(nTokens) + " tokens but --nbatches is " +
                std::to_string(args->nBatches), "questions");
        }
        if ((NnUint)nTokens > header->seqLen)
            throw System1RequestException("the prompt exceeds the model context length", "questions");

        inference->setBatchSize((NnUint)nTokens);
        inference->setPosition(0u);
        for (int i = 0; i < nTokens; i++)
            inference->setToken((NnUint)i, (NnUint)tokens[i]);
        const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        inference->forward();

        float *lastLogits = &inference->outputPipe[((NnUint)nTokens - 1u) * header->vocabSize];
        std::vector<float> logits(row.options.size());
        for (size_t i = 0; i < row.options.size(); i++)
            logits[i] = lastLogits[decisionConfig.letterTokenIds[i]];

        Readout result;
        result.probabilities = softmax(logits, temperatureForType(row.type));
        result.inputTokens = (NnUint)nTokens;
        result.milliseconds = elapsedMilliseconds(started);
        return result;
    }

    System1Json answerQuestion(
        const System1Json &state, const System1Question &question,
        bool allowNone, bool noneCheck, bool details, NnUint *inputTokens,
        double *firstPassMilliseconds
    ) {
        const std::chrono::steady_clock::time_point questionStarted = std::chrono::steady_clock::now();
        const bool useNone = question.type == "choice" && (allowNone || noneCheck);
        DecisionRow row = buildRow(state, question, false);
        if (row.type == "choice" && row.options.size() == 1u && !useNone) {
            System1Json probabilities = System1Json::object();
            probabilities[row.options[0].id] = 1.0;
            System1Json answer = System1Json{
                {"type", "choice"}, {"choice", row.options[0].id},
                {"confidence", 1.0}, {"probabilities", probabilities},
            };
            if (details)
                answer["details"] = instantDetails(
                    probabilities, 1.0f, 0.0, elapsedMilliseconds(questionStarted));
            return answer;
        }
        if (row.type == "choice" && row.options.size() + (useNone ? 1u : 0u) > decisionConfig.maxOptions)
            return answerWideChoice(
                state, question, row.options, allowNone, useNone, details, inputTokens,
                firstPassMilliseconds, questionStarted);

        row = buildRow(state, question, useNone);
        const Readout result = read(row);
        *inputTokens += result.inputTokens;
        *firstPassMilliseconds += result.milliseconds;
        System1Json answer = buildAnswer(row, question, result.probabilities, allowNone);
        if (details) {
            answer["details"] = instantDetails(
                keyedProbabilities(row, result.probabilities),
                *std::max_element(result.probabilities.begin(), result.probabilities.end()),
                result.milliseconds, elapsedMilliseconds(questionStarted));
        }
        return answer;
    }

    System1Json buildAnswer(
        const DecisionRow &row, const System1Question &question,
        const std::vector<float> &rawProbabilities, bool allowNone
    ) const {
        if (row.type == "noul")
            return System1Json{{"type", "noul"}, {"noul", rawProbabilities[0]}};

        if (row.type == "score") {
            double score = 0.0;
            System1Json probabilities = System1Json::object();
            System1Json legend = System1Json::object();
            std::vector<System1Json> levels;
            if (question.criteria.is_array()) {
                for (size_t i = 0; i < question.criteria.size(); i++)
                    levels.push_back(question.criteria[i]);
            } else if (question.criteria.is_object()) {
                const std::vector<std::pair<std::string, System1Json>> ordered =
                    orderedScoreLevels(question.criteria);
                for (size_t i = 0; i < ordered.size(); i++)
                    levels.push_back(ordered[i].second);
            }
            for (size_t i = 0; i < rawProbabilities.size(); i++) {
                score += i * rawProbabilities[i];
                probabilities[std::to_string(i)] = rawProbabilities[i];
                legend[std::to_string(i)] = i < levels.size()
                    ? levels[i]
                    : System1Json(row.options[i].id);
            }
            return System1Json{
                {"type", "score"}, {"score", score},
                {"confidence", scoreConfidence(rawProbabilities)},
                {"legend", legend}, {"probabilities", probabilities},
            };
        }

        std::vector<float> probabilities = rawProbabilities;
        float noneProbability = -1.0f;
        bool abstain = false;
        if (row.hasNoneOption) {
            noneProbability = probabilities.back();
            probabilities.pop_back();
            abstain = allowNone && noneProbability > *std::max_element(probabilities.begin(), probabilities.end());
            float sum = 0.0f;
            for (size_t i = 0; i < probabilities.size(); i++) sum += probabilities[i];
            for (size_t i = 0; i < probabilities.size(); i++) probabilities[i] /= sum;
        }
        System1Json distribution = System1Json::object();
        for (size_t i = 0; i < probabilities.size(); i++)
            distribution[row.options[i].id] = probabilities[i];
        System1Json answer = {
            {"type", "choice"},
            {"choice", abstain ? System1Json(nullptr) : System1Json(row.options[maximumIndex(probabilities)].id)},
            {"confidence", choiceConfidence(probabilities)},
            {"probabilities", distribution},
        };
        if (noneProbability >= 0.0f) answer["none_probability"] = noneProbability;
        return answer;
    }

    System1Json answerWideChoice(
        const System1Json &state, const System1Question &question,
        const std::vector<DecisionOption> &options, bool allowNone, bool useNone,
        bool details, NnUint *inputTokens, double *firstPassMilliseconds,
        const std::chrono::steady_clock::time_point &questionStarted
    ) {
        const size_t nGroups = (options.size() + decisionConfig.wideGroup - 1u) / decisionConfig.wideGroup;
        const size_t groupSize = options.size() / nGroups;
        const size_t extra = options.size() % nGroups;
        std::vector<float> firstProbabilities(options.size());
        std::vector<size_t> finalists;
        size_t start = 0u;

        for (size_t groupIndex = 0; groupIndex < nGroups; groupIndex++) {
            const size_t end = start + groupSize + (groupIndex < extra ? 1u : 0u);
            const std::vector<DecisionOption> group(options.begin() + start, options.begin() + end);
            const DecisionRow groupRow = buildRow(state, question, false, &group);
            const Readout groupReadout = read(groupRow);
            *inputTokens += groupReadout.inputTokens;
            *firstPassMilliseconds += groupReadout.milliseconds;
            std::vector<size_t> indexes;
            for (size_t i = start; i < end; i++) {
                firstProbabilities[i] = groupReadout.probabilities[i - start];
                indexes.push_back(i);
            }
            std::stable_sort(indexes.begin(), indexes.end(),
                [&firstProbabilities](size_t a, size_t b) {
                    return firstProbabilities[a] > firstProbabilities[b];
                });
            const size_t keep = std::min((size_t)decisionConfig.wideKeep, indexes.size());
            finalists.insert(finalists.end(), indexes.begin(), indexes.begin() + keep);
            start = end;
        }

        const size_t maxFinalists = decisionConfig.maxOptions - (useNone ? 1u : 0u);
        if (finalists.size() > maxFinalists) {
            std::stable_sort(finalists.begin(), finalists.end(),
                [&firstProbabilities](size_t a, size_t b) {
                    return firstProbabilities[a] > firstProbabilities[b];
                });
            finalists.resize(maxFinalists);
        }

        std::vector<DecisionOption> finalOptions;
        for (size_t i = 0; i < finalists.size(); i++) finalOptions.push_back(options[finalists[i]]);
        const DecisionRow finalRow = buildRow(state, question, useNone, &finalOptions);
        const Readout finalReadout = read(finalRow);
        *inputTokens += finalReadout.inputTokens;
        *firstPassMilliseconds += finalReadout.milliseconds;

        std::vector<float> finalProbabilities = finalReadout.probabilities;
        float noneProbability = -1.0f;
        bool abstain = false;
        if (useNone) {
            noneProbability = finalProbabilities.back();
            finalProbabilities.pop_back();
            abstain = allowNone && noneProbability > *std::max_element(
                finalProbabilities.begin(), finalProbabilities.end());
            float finalSum = 0.0f;
            for (size_t i = 0; i < finalProbabilities.size(); i++) finalSum += finalProbabilities[i];
            for (size_t i = 0; i < finalProbabilities.size(); i++) finalProbabilities[i] /= finalSum;
        }

        std::vector<float> combined(options.size(), 0.0f);
        std::set<size_t> finalSet(finalists.begin(), finalists.end());
        for (size_t i = 0; i < finalists.size(); i++)
            combined[finalists[i]] = finalProbabilities[i] * (1.0f - decisionConfig.wideResidual);
        float restMass = 0.0f;
        for (size_t i = 0; i < options.size(); i++) {
            if (finalSet.count(i) == 0u) restMass += firstProbabilities[i];
        }
        if (restMass == 0.0f) restMass = 1.0f;
        for (size_t i = 0; i < options.size(); i++) {
            if (finalSet.count(i) == 0u)
                combined[i] = decisionConfig.wideResidual * firstProbabilities[i] / restMass;
        }
        float combinedSum = 0.0f;
        for (size_t i = 0; i < combined.size(); i++) combinedSum += combined[i];
        for (size_t i = 0; i < combined.size(); i++) combined[i] /= combinedSum;

        System1Json distribution = System1Json::object();
        for (size_t i = 0; i < options.size(); i++) distribution[options[i].id] = combined[i];
        System1Json answer = {
            {"type", "choice"},
            {"choice", abstain ? System1Json(nullptr) : System1Json(options[maximumIndex(combined)].id)},
            {"confidence", choiceConfidence(combined)},
            {"probabilities", distribution},
        };
        if (noneProbability >= 0.0f) answer["none_probability"] = noneProbability;
        if (details) {
            System1Json first = System1Json::object();
            if (useNone) {
                for (size_t i = 0; i < options.size(); i++)
                    first[options[i].id] = combined[i] * (1.0f - noneProbability);
                first[NONE_OPTION] = noneProbability;
            } else {
                for (size_t i = 0; i < options.size(); i++)
                    first[options[i].id] = combined[i];
            }
            answer["details"] = instantDetails(
                first,
                *std::max_element(finalReadout.probabilities.begin(), finalReadout.probabilities.end()),
                *firstPassMilliseconds, elapsedMilliseconds(questionStarted));
        }
        return answer;
    }
};

static float configTemperature(const System1Json &temperatures, const char *type) {
    if (!temperatures.contains(type)) return 1.0f;
    if (!temperatures[type].is_number())
        throw std::runtime_error(std::string("temperature_by_type.") + type + " must be a number");
    const float value = temperatures[type].get<float>();
    if (!std::isfinite(value) || value <= 0.0f)
        throw std::runtime_error(std::string("temperature_by_type.") + type + " must be positive");
    return value;
}

static System1DecisionConfig loadDecisionConfig(const std::string &path) {
    std::ifstream file(path.c_str());
    if (!file)
        throw std::runtime_error("Cannot open decision config: " + path);
    System1Json root;
    file >> root;
    if (root.value("format", std::string()) != "lodestar-letter-readout-v1")
        throw std::runtime_error("Unsupported decision config format");
    if (root.value("prompt_attention", std::string()) != "bidirectional")
        throw std::runtime_error("The decision config is not for bidirectional Instant inference");
    if (!root.contains("letter_token_ids") || !root["letter_token_ids"].is_array())
        throw std::runtime_error("decision_config.json lacks letter_token_ids");

    System1DecisionConfig config;
    for (size_t i = 0; i < root["letter_token_ids"].size(); i++) {
        if (!root["letter_token_ids"][i].is_number_unsigned() &&
            !root["letter_token_ids"][i].is_number_integer())
            throw std::runtime_error("letter_token_ids must contain integers");
        config.letterTokenIds.push_back(root["letter_token_ids"][i].get<NnUint>());
    }
    const System1Json temperatures = root.value("temperature_by_type", System1Json::object());
    config.choiceTemperature = configTemperature(temperatures, "choice");
    config.noulTemperature = configTemperature(temperatures, "noul");
    config.scoreTemperature = configTemperature(temperatures, "score");
    config.maxOptions = root.value("max_options_per_pass", 26u);
    if (config.maxOptions != N_LETTERS)
        throw std::runtime_error("Only max_options_per_pass=26 is supported");
    if (root.contains("wide_choice")) {
        const System1Json &wide = root["wide_choice"];
        config.wideGroup = wide.value("group", 25u);
        config.wideKeep = wide.value("keep", 3u);
        config.wideResidual = wide.value("residual", 0.001f);
    }
    if (config.wideGroup == 0u || config.wideGroup >= config.maxOptions || config.wideKeep == 0u ||
        config.wideKeep > config.wideGroup || !std::isfinite(config.wideResidual) ||
        config.wideResidual < 0.0f || config.wideResidual >= 1.0f)
        throw std::runtime_error("Invalid wide_choice configuration");
    config.releaseDate = root.value("release_date", std::string());
    return config;
}

static void handleDecision(HttpRequest &request, System1Api *api) {
    api->decide(request);
}

static void handleModels(HttpRequest &request, System1Api *api) {
    api->models(request);
}

static void handleHealth(HttpRequest &request, System1Api *api) {
    api->health(request);
}

static void server(AppInferenceContext *context) {
    System1Api api(context);
    if (std::strcmp(context->args->host, "0.0.0.0") == 0 ||
        std::strcmp(context->args->host, "127.0.0.1") == 0)
        printf("Server URL: http://localhost:%u/v1/systemone\n", context->args->port);

    std::vector<Route> routes = {
        {"/v1/systemone", HttpMethod::METHOD_POST,
            std::bind(&handleDecision, std::placeholders::_1, &api)},
        {"/v1/decisions", HttpMethod::METHOD_POST,
            std::bind(&handleDecision, std::placeholders::_1, &api)},
        {"/v1/models", HttpMethod::METHOD_GET,
            std::bind(&handleModels, std::placeholders::_1, &api)},
        {"/health", HttpMethod::METHOD_GET,
            std::bind(&handleHealth, std::placeholders::_1, &api)},
        {"/v1/health", HttpMethod::METHOD_GET,
            std::bind(&handleHealth, std::placeholders::_1, &api)},
    };
    runHttpServer(context->args->host, context->args->port, routes);
}

static void usage() {
    fprintf(stderr, "Usage: systemone-api --model <path> --tokenizer <path> --decision-config <path> [options]\n");
    fprintf(stderr, "        [--nbatches <maximum prompt tokens>] [--max-seq-len <max>]\n");
    fprintf(stderr, "        [--buffer-float-type {f32|f16|q40|q80}] [--nthreads <n>]\n");
    fprintf(stderr, "        [--host <addr>] [--port <port>] [--workers <ip:port> ...]\n");
}

int main(int argc, char *argv[]) {
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif
    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, false, "--decision-config");
        if (args.help) {
            usage();
            return EXIT_SUCCESS;
        }
        if (args.extraConfigPath == nullptr)
            throw std::runtime_error("--decision-config is required");
        decisionConfig = loadDecisionConfig(args.extraConfigPath);

        initQuants();
        initSockets();
        runInferenceApp(&args, server);
        cleanupSockets();
    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
