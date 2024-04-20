#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <sys/mman.h>
#include <ctype.h>
#include <ctime>
#include <cassert>
#include "funcs.hpp"
#include "utils.hpp"
#include "tokenizer.hpp"

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

bool isSafePiece(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) return false;
    if (piece[0] == '\0') return false;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return false; // bad byte, don't print it
        }
    }
    return true;
}

void safePrintf(char *piece) {
    if (isSafePiece(piece)) {
        printf("%s", piece);
    }
}

Tokenizer::Tokenizer(char* tokenizerPath, int vocab_size, bool bos, bool eos) {
    // i should have written the vocab_size into the tokenizer file... sigh
    this->vocab_size = vocab_size;
    this->bos = bos;
    this->eos = eos;
    // malloc space to hold the scores and the strings
    vocab = (char**)malloc(vocab_size * sizeof(char*));
    vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        byte_pieces[i * 2] = (unsigned char)i;
        byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizerPath, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open %s\n", tokenizerPath);
        exit(EXIT_FAILURE);
    }
    if (fread(&max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        vocab[i] = (char *)malloc(len + 1);
        if (fread(vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

Tokenizer::~Tokenizer() {
    for (int i = 0; i < vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    free(vocab_scores);
    free(sorted_vocab);
}

char* Tokenizer::decode(int prev_token, int token) {
    char *piece = vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)byte_pieces + byte_val * 2;
    }
    return piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void Tokenizer::encode(char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        sorted_vocab = new TokenIndex[vocab_size];
        for (int i = 0; i < vocab_size; i++) {
            sorted_vocab[i].str = vocab[i];
            sorted_vocab[i].id = i;
        }
        qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    int str_buffer_size = max_token_length*2 +1 +2;
    char* str_buffer = new char[str_buffer_size];
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        char space[] = " ";
        int dummy_prefix = str_lookup(space, sorted_vocab, vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            snprintf(str_buffer, str_buffer_size, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, sorted_vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void readStdin(const char* guide, char* buffer, size_t bufsize) {
    fflush(stdin);
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

Sampler::Sampler(int vocab_size, float temperature, float topp, unsigned long long rngSeed) {
    this->vocab_size = vocab_size;
    this->temperature = temperature;
    this->topp = topp;
    this->rngState = rngSeed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    probindex = new ProbIndex[vocab_size];
}

Sampler::~Sampler() {
    free(probindex);
}

int Sampler::sample(float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q < vocab_size; q++) { logits[q] /= temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = randomF32(&rngState);
        // we sample from this distribution to get the next token
        if (topp <= 0 || topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, vocab_size, topp, probindex, coin);
        }
    }
    return next;
}
