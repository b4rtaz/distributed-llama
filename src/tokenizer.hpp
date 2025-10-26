#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <cstdio>
#include <string>
#include <vector>

typedef struct {
    char *str;
    unsigned int id;
} TokenIndex;

struct TokenizerOldHeader {
    unsigned int vocabSize;
    unsigned int maxTokenLength;
    int bosId;
    int eosId;
    int padId;
};

enum TokenizerHeaderKey {
    TOK_VERSION = 0,
    TOK_VOCAB_SIZE = 1,
    MAX_TOKEN_LENGTH = 2,
    BOS_ID = 3,
    EOS_ID = 4, // Backward compatibility
    PAD_ID = 5, // Ignored
    CHAT_EOS_ID = 6, // Backward compatibility
    CHAT_TEMPLATE = 7,
    CHAT_STOP = 8, // Ignored
    N_EOS_TOKENS = 9,
    ADD_BOS = 10,
};

class Tokenizer {
private:
    unsigned int maxTokenLength;
    unsigned int regularVocabSize;
    unsigned int specialVocabSize;
    float *vocabScores;
    unsigned int *vocabLength;
    TokenIndex *regularVocab;
    TokenIndex *specialVocab;
    size_t strBufferSize;
    char *strBuffer;
    char *utf8Buffer;
    size_t strBufferPos;


public:
    std::vector<int> eosTokenIds;
    unsigned int vocabSize;
    char **vocab;
    int bosId;
    bool addBos;
    char *chatTemplate;

    Tokenizer(const char *tokenizer_path);
    ~Tokenizer();
    void printHeader();
    int findSpecialTokenStartWith(char *piece);
    int findRegularToken(char *piece);
    void encode(char *text, int *tokens, int *nTokens, bool isStart, bool addSpecialTokens);
    bool isEos(int token);
    char *decode(int token);
    void resetDecoder();

private:
    char *detokUtf8();
};

typedef struct {
    float prob;
    int index;
} ProbIndex;

class Sampler {
private:
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rngState;

public:
    Sampler(int vocab_size, float temperature, float topp, unsigned long long rngSeed);
    ~Sampler();
    int sample(float *logits);
    void setTemp(float temp);
    void setSeed(unsigned long long rngSeed);
};

class TokenizerChatStops {
public:
    const char **stops;
    size_t nStops;
    size_t maxStopLength;
    TokenizerChatStops(Tokenizer *tokenizer);
    ~TokenizerChatStops();
};

enum ChatTemplateType {
    TEMPLATE_UNKNOWN = 0,
    TEMPLATE_LLAMA2 = 1,
    TEMPLATE_LLAMA3 = 2,
    TEMPLATE_DEEP_SEEK3 = 3,
    TEMPLATE_CHATML = 4,
};

struct ChatItem {
    std::string role;
    std::string message;
};

struct GeneratedChat {
    const char *content;
    size_t length;
    const char *publicPrompt;
};

class ChatTemplateGenerator {
public:
    const char *eos;
    ChatTemplateType type;
    std::string buffer;
    ChatTemplateGenerator(const ChatTemplateType type, const char *chatTemplate, const char *eos);
    GeneratedChat generate(unsigned int nItems, ChatItem *items, bool appendGenerationPrompt);
};

enum EosDetectorType {
    MAYBE_EOS = 0,
    EOS = 1,
    NOT_EOS = 2,
};

class EosDetector {
private:
    size_t nTokens;
    const int *tokens;
    const char **pieces;
    size_t *pieceSizes;
    size_t bufferPos;
    size_t bufferSize;
    int eosPos;
    int paddingLeft;
    int paddingRight;
public:
    char *buffer;
    EosDetector(size_t nTokens, const int *tokens, const char* *pieces, int paddingLeft, int paddingRight);
    ~EosDetector();

    EosDetectorType append(int tokenId, const char *piece);
    bool isEos(int tokenId);
    char *getDelta();
    void reset();
};

#endif
