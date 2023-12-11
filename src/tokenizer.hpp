#include <stdio.h>

void safePrintf(char *piece);

typedef struct {
    char *str;
    int id;
} TokenIndex;

class Tokenizer {
private:
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings

public:
    Tokenizer(char* tokenizer_path, int vocab_size);
    ~Tokenizer();
    void encode(char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
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
    unsigned long long rng_state;

public:
    Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed);
    ~Sampler();
    int sample(float* logits);
};
