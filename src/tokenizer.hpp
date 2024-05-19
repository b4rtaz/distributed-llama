#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <cstdio>
#include "tasks.hpp"

bool isSafePiece(char *piece);
void safePrintf(char *piece);
void readStdin(const char* guide, char* buffer, size_t bufsize);

typedef struct {
    char *str;
    int id;
} TokenIndex;

struct TokenizerHeader {
    unsigned int magic;
    unsigned int vocabSize;
    unsigned int maxTokenLength;
    int bosId;
    int eosId;
    int padId;
};

class Tokenizer {
private:
    unsigned int maxTokenLength;
    char** vocab;
    float* vocabScores;
    TokenIndex *sortedVocab;
    int vocabSize;
    unsigned char bytePieces[512]; // stores all single-byte strings

public:
    int bosId;
    int eosId;

    Tokenizer(char* tokenizer_path, int vocab_size);
    ~Tokenizer();
    void encode(char *text, int *tokens, int *nTokens, bool addBos, bool addEos);
    char* decode(int prev_token, int token);
};

// struct used when sorting probabilities during top-p sampling
typedef struct {
    float prob;
    int index;
} ProbIndex;

// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
class Sampler {
private:
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rngState;

public:
    Sampler(int vocab_size, float temperature, float topp, unsigned long long rngSeed);
    ~Sampler();
    int sample(float* logits);
    void setTemp(float temp);
    void setSeed(unsigned long long rngSeed);
};

#endif
