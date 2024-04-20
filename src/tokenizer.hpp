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

class Tokenizer {
private:
    bool bos;
    bool eos;
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings

public:
    Tokenizer(char* tokenizer_path, int vocab_size, bool bos, bool eos);
    ~Tokenizer();
    void encode(char *text, int *tokens, int *n_tokens);
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
};

#endif
