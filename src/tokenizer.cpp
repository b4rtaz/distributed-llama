#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <ctype.h>
#include <ctime>
#include <cassert>
#include <stdexcept>
#include <sstream>
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

Tokenizer::Tokenizer(char* tokenizerPath, int modelVocabSize) {
    eosId = -1;
    bosId = -1;
    chatEosId = -1;
    chatTemplate = NULL;
    chatStop = NULL;

    // read in the file
    FILE *file = fopen(tokenizerPath, "rb");
    if (!file)
        throw std::runtime_error("Failed to open tokenizer file");
    int magic;
    if (fread(&magic, sizeof(int), 1, file) != 1)
        throw std::runtime_error("Cannot read tokenizer magic number");

    if (magic == 0x567123) {
        TokenizerOldHeader header;
        if (fread(&header, sizeof(TokenizerOldHeader), 1, file) != 1)
            throw std::runtime_error("Cannot read tokenizer header");
        maxTokenLength = header.maxTokenLength;
        vocabSize = header.vocabSize;
        bosId = header.bosId;
        eosId = header.eosId;
    } else if (magic == 0x567124) {
        TransformerHeaderKey key;
        int headerSize;
        if (fread(&headerSize, sizeof(int), 1, file) != 1)
            throw std::runtime_error("Cannot read tokenizer header size");
        int nKv = (headerSize - 2 * sizeof(int)) / sizeof(int);
        int buffer[nKv];
        if (fread(&buffer, nKv * sizeof(int), 1, file) != 1) {
            throw std::runtime_error("Cannot read header values");
        }
        int version = -1;
        int chatTemplateLength = -1;
        int chatStopLength = -1;
        for (int i = 0; i < nKv; i += 2) {
            int key = buffer[i];
            int value = buffer[i + 1];
            if (key == TOK_VERSION) version = value;
            else if (key == TOK_VOCAB_SIZE) vocabSize = value;
            else if (key == MAX_TOKEN_LENGTH) maxTokenLength = (unsigned int)value;
            else if (key == BOS_ID) bosId = value;
            else if (key == EOS_ID) eosId = value;
            else if (key == CHAT_EOS_ID) chatEosId = value;
            else if (key == CHAT_TEMPLATE) chatTemplateLength = value;
            else if (key == CHAT_STOP) chatStopLength = value;
            else if (key == PAD_ID) {} // ignore
            else {
                throw std::runtime_error("Invalid tokenizer header key:" + std::to_string(key));
            }
        }

        if (version != 1)
            throw std::runtime_error("Old tokenizer version, please regenerate your tokenizer");

        if (chatTemplateLength > 0) {
            chatTemplate = new char[chatTemplateLength];
            if (fread(chatTemplate, chatTemplateLength, 1, file) != 1)
                throw std::runtime_error("Cannot read chat template from tokenizer file");
        }
        if (chatStopLength > 0) {
            chatStop = new char[chatStopLength];
            if (fread(chatStop, chatStopLength, 1, file) != 1)
                throw std::runtime_error("Cannot read chat stop from tokenizer file");
        }
    } else {
        throw std::runtime_error("Invalid tokenizer file");
    }

    if (maxTokenLength < 1 || vocabSize != modelVocabSize) {
        throw std::runtime_error("Tokenizer file is invalid or incompatible with model");
    }

    if (bosId >= 0) printf("📄 bosId: %d\n", bosId);
    if (eosId >= 0) printf("📄 eosId: %d\n", eosId);
    if (chatEosId >= 0) printf("📄 chatEosId: %d\n", chatEosId);

    // malloc space to hold the scores and the strings
    vocab = (char**)malloc(vocabSize * sizeof(char*));
    vocabScores = (float*)malloc(vocabSize * sizeof(float));
    sortedVocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        bytePieces[i * 2] = (unsigned char)i;
        bytePieces[i * 2 + 1] = '\0';
    }

    int len;
    for (int i = 0; i < vocabSize; i++) {
        if (fread(vocabScores + i, sizeof(float), 1, file) != 1)
            throw std::runtime_error("Cannot read size from tokenizer file");
        if (fread(&len, sizeof(int), 1, file) != 1)
            throw std::runtime_error("Cannot read length from tokenizer file");
        vocab[i] = (char *)malloc(len + 1);
        if (fread(vocab[i], len, 1, file) != 1)
            throw std::runtime_error("Cannot read word from tokenizer file");
        vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

Tokenizer::~Tokenizer() {
    if (chatTemplate != NULL) delete[] chatTemplate;
    if (chatStop != NULL) delete[] chatStop;

    for (int i = 0; i < vocabSize; i++) { free(vocab[i]); }
    free(vocab);
    free(vocabScores);
    free(sortedVocab);
}

char* Tokenizer::decode(int prev_token, int token) {
    char *piece = vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == bosId && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == bosId) {
        piece = (char*)bytePieces + byte_val * 2;
    }
    return piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void Tokenizer::encode(char *text, int *tokens, int *nTokens, bool addBos, bool addEos) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (sortedVocab == NULL) {
        // lazily malloc and sort the vocabulary
        sortedVocab = new TokenIndex[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            sortedVocab[i].str = vocab[i];
            sortedVocab[i].id = i;
        }
        qsort(sortedVocab, vocabSize, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    int str_buffer_size = maxTokenLength*2 +1 +2;
    char* str_buffer = new char[str_buffer_size];
    size_t str_len = 0;

    // start at 0 tokens
    *nTokens = 0;

    // add optional BOS (=1) token, if desired
    if (addBos) tokens[(*nTokens)++] = bosId;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        char space[] = " ";
        int dummy_prefix = str_lookup(space, sortedVocab, vocabSize);
        // TODO: this condition saves us from segmentation fault
        if (dummy_prefix != -1)
            tokens[(*nTokens)++] = dummy_prefix;
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
        int id = str_lookup(str_buffer, sortedVocab, vocabSize);
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*nTokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*nTokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*nTokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            snprintf(str_buffer, str_buffer_size, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, sortedVocab, vocabSize);
            if (id != -1 && vocabScores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocabScores[id];
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
        for (int i = best_idx+1; i < (*nTokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*nTokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (addEos) tokens[(*nTokens)++] = eosId;

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

void Sampler::setTemp(float temp) {
    this->temperature = temp;
}

void Sampler::setSeed(unsigned long long seed) {
    this->rngState = seed;
}

TokenizerChatStops::TokenizerChatStops(Tokenizer* tokenizer) {
    const bool hasExtraStop = tokenizer->chatStop != NULL;
    nStops = hasExtraStop ? 2 : 1;
    char** s = new char*[nStops];
    s[0] = tokenizer->vocab[tokenizer->chatEosId];
    if (hasExtraStop)
        s[1] = tokenizer->chatStop;
    maxStopLength = 0;
    for (size_t i = 0; i < nStops; i++) {
        size_t len = strlen(s[i]);
        if (len > maxStopLength) maxStopLength = len;
    }
    stops = (const char**)s;
}

TokenizerChatStops::~TokenizerChatStops() {
    delete[] stops;
}

ChatTemplate::ChatTemplate(const ChatTemplateType type, const char* chatTemplate, const char* eos) {
    if (type == TEMPLATE_UNKNOWN) {
        if (chatTemplate == NULL)
            throw std::runtime_error("The tokenizer does not include chat template");
        if (strstr(chatTemplate, "[INST]") != NULL) {
            this->type = TEMPLATE_LLAMA2;
        } else if (strstr(chatTemplate, "<|start_header_id|>") != NULL) {
            this->type = TEMPLATE_LLAMA3;
        } else if (strstr(chatTemplate, "<|user|>") != NULL) {
            this->type = TEMPLATE_ZEPHYR;
        } else if (strstr(chatTemplate, "<|im_start|>") != NULL) {
            this->type = TEMPLATE_CHATML;
        } else {
            throw new std::runtime_error("Not supported chat template");
        }
    } else {
        this->type = type;
    }
    this->eos = eos;

    printf("⭐ chat template: ");
    if (this->type == TEMPLATE_LLAMA2) {
        printf("llama2\n");
    } else if (this->type == TEMPLATE_LLAMA3) {
        printf("llama3\n");
    } else if (this->type == TEMPLATE_ZEPHYR) {
        printf("zephyr\n");
    } else if (this->type == TEMPLATE_CHATML) {
        printf("chatml\n");
    }
}

std::string ChatTemplate::generate(unsigned int nMessages, ChatItem* items, bool appendGenerationPrompt) {
    std::ostringstream buffer;
    if (type == TEMPLATE_LLAMA2) {
        unsigned int i = 0;
        if (nMessages >= 2 && items[0].role == "system" && items[1].role == "user") {
            buffer << "[INST] <<SYS>>\n" << items[0].message << "\n<</SYS>>\n\n" << items[1].message << " [/INST]" << eos;
            i += 2;
        }
        for (; i < nMessages; i++) {
            if (items[i].role == "assistant") {
                buffer << items[i].message << eos;
            } else if (items[i].role == "user") {
                buffer << "[INST] " << items[i].message << " [/INST]" << eos;
            }
        }
    } else if (type == TEMPLATE_LLAMA3) {
        for (unsigned int i = 0; i < nMessages; i++)
            buffer << "<|start_header_id|>" << items[i].role << "<|end_header_id|>\n\n" << items[i].message << eos;
        if (appendGenerationPrompt)
            buffer << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (type == TEMPLATE_CHATML) {
        for (unsigned int i = 0; i < nMessages; i++)
            buffer << "<|im_start|>" << items[i].role << "\n" << items[i].message << "<|im_end|>\n";
        if (appendGenerationPrompt)
            buffer << "<|im_start|>assistant\n";
    } else if (type == TEMPLATE_ZEPHYR) {
        for (unsigned int i = 0; i < nMessages; i++)
            buffer << "<|" << items[i].role << "|>\n" << items[i].message << eos << "\n";
        if (appendGenerationPrompt)
            buffer << "<|assistant|>\n";
    }
    return buffer.str();
}

EosDetector::EosDetector(int eosId, size_t nStops, const char** stops, int paddingLeft, int paddingRight) {
    this->eosId = eosId;
    this->nStops = nStops;
    this->stops = stops;
    this->stopSizes = new size_t[nStops];
    for (size_t s = 0; s < nStops; s++) {
        stopSizes[s] = strlen(stops[s]);
        printf("🛑 stop: %s\n", stops[s]);
    }
    this->bufferPos = 0;
    this->bufferSize = 0;
    this->paddingLeft = paddingLeft;
    this->paddingRight = paddingRight;
}

EosDetector::~EosDetector() {
    if (bufferSize > 0)
        delete[] buffer;
    delete[] stopSizes;
}

EosDetectorType EosDetector::append(int tokenId, const char* piece) {
    int pieceLength = strlen(piece);
    int newSize = bufferPos + pieceLength + 1;
    if (newSize > bufferSize) {
        char* newBuffer = new char[newSize];
        if (bufferPos > 0)
            memcpy(newBuffer, buffer, bufferPos);
        if (bufferSize > 0)
            delete[] buffer;
        buffer = newBuffer;
        bufferSize = newSize;
    }
    memcpy(&buffer[bufferPos], piece, pieceLength + 1);
    bufferPos += pieceLength;

    // detection

    if (tokenId == eosId) {
        eosPos = bufferPos - pieceLength;
        return EOS;
    }
    eosPos = -1;

    for (size_t s = 0; s < nStops; s++) {
        size_t stopSize = stopSizes[s];
        if (bufferPos > stopSize + paddingLeft + paddingRight) continue;

        for (int lo = 0; lo <= paddingLeft; lo++) {
            int n = bufferPos - lo;
            if (n == 0 || n > stopSize + paddingRight) continue;
            if (n > stopSize) n = stopSize;
            if (strncmp(buffer + lo, stops[s], n) == 0) {
                if (n == stopSize) {
                    eosPos = lo;
                    return EOS;
                }
                return MAYBE_EOS;
            }
        }
    }
    return NOT_EOS;
}

char* EosDetector::getDelta() {
    if (eosPos == -1) return buffer;
    if (eosPos == 0) return NULL;
    buffer[eosPos] = '\0';
    return buffer;
}

void EosDetector::clear() {
    bufferPos = 0;
}
