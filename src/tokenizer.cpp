#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <ctype.h>
#include <ctime>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <vector>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu-ops.hpp"
#include "tokenizer.hpp"
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

#define DEBUG_TOKENIZER_ENCODER false
#define DEBUG_TOKENIZER_BENCHMARK false
#define DEBUG_TEMPLATE_GENERATOR false
#define DEBUG_SAMPLER_BENCHMARK false

unsigned int randomU32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float randomF32(unsigned long long *state) {
    // random float32 in <0,1)
    return (randomU32(state) >> 8) / 16777216.0f;
}

int compareTokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

Tokenizer::Tokenizer(const char* tokenizerPath)
    : eosTokenIds() {
    bosId = -1;
    chatTemplate = nullptr;
    maxTokenLength = 0;

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
        eosTokenIds.push_back(header.eosId);
    } else if (magic == 0x567124) {
        int headerSize;
        if (fread(&headerSize, sizeof(int), 1, file) != 1)
            throw std::runtime_error("Cannot read tokenizer header size");
        int nKv = (headerSize - 2 * sizeof(int)) / sizeof(int);
        std::vector<int> buffer(nKv);
        if (fread(&buffer[0], nKv * sizeof(int), 1, file) != 1) {
            throw std::runtime_error("Cannot read header values");
        }
        int version = -1;
        int chatTemplateLength = -1;
        int nEosTokens = 0;
        for (int i = 0; i < nKv; i += 2) {
            int key = buffer[i];
            int value = buffer[i + 1];
            if (key == TOK_VERSION) version = value;
            else if (key == TOK_VOCAB_SIZE) vocabSize = value;
            else if (key == MAX_TOKEN_LENGTH) maxTokenLength = (unsigned int)value;
            else if (key == BOS_ID) bosId = value;
            else if (key == EOS_ID) eosTokenIds.push_back(value); // Backward compatibility
            else if (key == CHAT_EOS_ID) eosTokenIds.push_back(value); // Backward compatibility
            else if (key == CHAT_TEMPLATE) chatTemplateLength = value;
            else if (key == CHAT_STOP) fseek(file, value, SEEK_CUR); // Ignored
            else if (key == PAD_ID) {} // Ignored
            else if (key == N_EOS_TOKENS) nEosTokens = value;
            else if (key == ADD_BOS) addBos = value == 1;
            else {
                throw std::runtime_error("Invalid tokenizer header key:" + std::to_string(key));
            }
        }

        if (version != 1)
            throw std::runtime_error("Old tokenizer version, please regenerate your tokenizer");

        if (chatTemplateLength > 0) {
            chatTemplate = new char[chatTemplateLength + 1];
            if (fread(chatTemplate, chatTemplateLength, 1, file) != 1)
                throw std::runtime_error("Cannot read chat template from tokenizer file");
            chatTemplate[chatTemplateLength] = '\0';
        }
        if (nEosTokens > 0) {
            int eosTokenId;
            for (int i = 0; i < nEosTokens; i++) {
                if (fread(&eosTokenId, sizeof(int), 1, file) != 1)
                    throw std::runtime_error("Cannot read eos token id from tokenizer file");
                eosTokenIds.push_back(eosTokenId);
            }
        }
    } else {
        throw std::runtime_error("Invalid tokenizer file");
    }

    if (maxTokenLength < 1)
        throw std::runtime_error("Invalid tokenizer max token length");

    // malloc space to hold the scores and the strings
    vocab = new char*[vocabSize];
    vocabLength = new unsigned int[vocabSize];
    vocabScores = new float[vocabSize];

    int length;
    for (int i = 0; i < vocabSize; i++) {
        if (fread(vocabScores + i, sizeof(float), 1, file) != 1)
            throw std::runtime_error("Cannot read size from tokenizer file");
        if (fread(&length, sizeof(int), 1, file) != 1)
            throw std::runtime_error("Cannot read length from tokenizer file");
        vocab[i] = new char[length + 1];
        if (fread(vocab[i], length, 1, file) != 1)
            throw std::runtime_error("Cannot read word from tokenizer file");
        vocab[i][length] = '\0'; // add the string terminating token
        vocabLength[i] = length;
    }

    // TODO: this is very unstable assumption that bosId splits regular and special vocab
    regularVocabSize = bosId;
    specialVocabSize = vocabSize - regularVocabSize;

    regularVocab = new TokenIndex[regularVocabSize];
    for (int i = 0; i < regularVocabSize; i++) {
        regularVocab[i].str = vocab[i];
        regularVocab[i].id = i;
    }
    qsort(regularVocab, regularVocabSize, sizeof(TokenIndex), compareTokens);

    specialVocab = new TokenIndex[specialVocabSize];
    for (int i = 0; i < specialVocabSize; i++) {
        specialVocab[i].str = vocab[i + regularVocabSize];
        specialVocab[i].id = i + regularVocabSize;
    }

    strBufferSize = maxTokenLength * 2;
    if (strBufferSize < (4 * 2)) { // ensure place for 2 utf-8 multi-byte sequence
        strBufferSize = 4 * 2;
    }
    strBufferSize += 1 + 2;
    strBuffer = new char[strBufferSize];
    utf8Buffer = new char[strBufferSize];

    fclose(file);
}

Tokenizer::~Tokenizer() {
    if (chatTemplate != NULL) delete[] chatTemplate;

    for (int i = 0; i < vocabSize; i++)
        delete[] vocab[i];
    delete[] vocab;
    delete[] vocabLength;
    delete[] vocabScores;
    delete[] regularVocab;
    delete[] specialVocab;
    delete[] strBuffer;
    delete[] utf8Buffer;
}

void Tokenizer::printHeader() {
    if (bosId >= 0) {
        printf("📄 AddBos: %d\n", addBos ? 1 : 0);
        printf("📄 BosId: %d (%s)\n", bosId, vocab[bosId]);
    }
    if (eosTokenIds.size() > 0) {
        printf("📄 EosId: ");
        for (unsigned int i = 0; i < eosTokenIds.size(); i++) {
            printf("%d (%s) ", eosTokenIds[i], vocab[eosTokenIds[i]]);
        }
        printf("\n");
    }
    printf("📄 RegularVocabSize: %d\n", regularVocabSize);
    printf("📄 SpecialVocabSize: %d\n", specialVocabSize);
}

int Tokenizer::findSpecialTokenStartWith(char *piece) {
    for (unsigned int i = 0; i < specialVocabSize; i++) {
        unsigned int tokenId = specialVocab[i].id;
        unsigned int length = vocabLength[tokenId];
        if (std::strncmp(vocab[tokenId], piece, length) == 0)
            return tokenId;
    }
    return -1;
}

int Tokenizer::findRegularToken(char *piece) {
    TokenIndex tok = { .str = piece };
    TokenIndex *res = (TokenIndex*)bsearch(&tok, regularVocab, regularVocabSize, sizeof(TokenIndex), compareTokens);
    return res != NULL ? res->id : -1;
}

bool Tokenizer::isEos(int token) {
    for (unsigned int i = 0; i < eosTokenIds.size(); i++) {
        if (token == eosTokenIds[i])
            return true;
    }
    return false;
}

void Tokenizer::resetDecoder() {
    strBufferPos = 0;
}

char *Tokenizer::detokUtf8() {
    char* src = strBuffer;
    char* dst = utf8Buffer;
    char* checkpoint_src = src;
    char* checkpoint = dst;
    unsigned expect_continuation = 0;

    while (unsigned char c = *src) {
        bool need_recovery = false;
        if (expect_continuation) {
            if ((c & 0xc0) == 0x80) {
                *dst++ = *src++;
                expect_continuation--;
            } else {
                need_recovery = true;
            }
        } else if (c <= 0x7f) {
            *dst++ = *src++;
        } else if (c >= 0xc0 && c <= 0xdf) {
            *dst++ = *src++;
            expect_continuation = 1;
        } else if (c >= 0xe0 && c <= 0xef) {
            *dst++ = *src++;
            expect_continuation = 2;
        } else if (c >= 0xf0 && c <= 0xf7) {
            *dst++ = *src++;
            expect_continuation = 3;
        } else {
            need_recovery = true;
        }

        if (!need_recovery) {
            if (!expect_continuation) {
                checkpoint = dst;
                checkpoint_src = src;
            }
        } else {
            // perform stream recover
            if (expect_continuation) {
                expect_continuation = 0;
            } else {
                ++src;
            }
            dst = checkpoint;
            // emit 0xfffd
            *dst++ = 0xef;
            *dst++ = 0xbf;
            *dst++ = 0xbd;

            fprintf(stderr, "Tokenizer: decoded invalid utf8 -- attempting stream recover\n");
        }
    }

    if (src > checkpoint_src) {
        memmove(strBuffer, checkpoint_src, src - checkpoint_src + 1);
        strBufferPos = src - checkpoint_src;
    } else {
        strBufferPos = 0;
    }
    *checkpoint = '\0';
    if (checkpoint > utf8Buffer) {
        return utf8Buffer;
    } else {
        return nullptr;
    }
}

char *Tokenizer::decode(int token) {
    if (token == bosId)
        return nullptr;
    if (isEos(token)) {
        if (strBufferPos > 0)
            return strBuffer;
        return nullptr;
    }

    char *piece = vocab[token];
    int pieceLen = vocabLength[token];

    assert(strBufferPos + pieceLen + 1 < strBufferSize);
    std::memcpy(&strBuffer[strBufferPos], piece, pieceLen * sizeof(char));
    strBufferPos += pieceLen;
    strBuffer[strBufferPos] = '\0';

    return detokUtf8();
}

void Tokenizer::encode(char *text, int *tokens, int *nTokens, bool isStart, bool addSpecialTokens) {
#if DEBUG_TOKENIZER_BENCHMARK
    Timer startTime;
#endif
    if (text == nullptr)
        throw std::runtime_error("Input text is null");

    size_t strLen = 0;

    *nTokens = 0;

    if (isStart && addBos && bosId >= 0)
        tokens[(*nTokens)++] = bosId;

    for (char *c = text; *c != '\0'; c++) {
        if (addSpecialTokens) {
            int specialTokenId = findSpecialTokenStartWith(c);
            if (specialTokenId >= 0) {
                tokens[(*nTokens)++] = specialTokenId;
                c += vocabLength[specialTokenId] - 1;
                continue;
            }
        }

        strBuffer[strLen] = *c;
        strLen++;
        assert(strLen < strBufferSize);
        strBuffer[strLen] = '\0';

        int id = findRegularToken(strBuffer);
        if (id != -1) {
            tokens[(*nTokens)++] = id;
            strLen = 0;
        }
    }

    assert(strLen == 0);

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*nTokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            snprintf(strBuffer, strBufferSize, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = findRegularToken(strBuffer);
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

#if DEBUG_TOKENIZER_BENCHMARK
    NnUint duration = startTime.elapsedMicroseconds();
    printf("🕒 [%22s] %u μs\n", "ENCODER", duration);
#endif
#if DEBUG_TOKENIZER_ENCODER
    printf("\033[1;33m[");
    for (unsigned int i = 0; i < *nTokens; i++)
        printf("%d,", tokens[i]);
    printf("]\033[0m");
#endif
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
    delete[] probindex;
}

int Sampler::sample(float* logits) {
#if DEBUG_SAMPLER_BENCHMARK
    Timer startTime;
#endif
    // sample the token given the logits and some hyperparameters
    int next;
    if (temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q < vocab_size; q++) { logits[q] /= temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax_F32(logits, vocab_size);
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
#if DEBUG_SAMPLER_BENCHMARK
    NnUint duration = startTime.elapsedMicroseconds();
    printf("🕒 [%22s] %u μs\n", "SAMPLER", duration);
#endif
    return next;
}

void Sampler::setTemp(float temp) {
    this->temperature = temp;
}

void Sampler::setSeed(unsigned long long seed) {
    this->rngState = seed;
}

TokenizerChatStops::TokenizerChatStops(Tokenizer* tokenizer) {
    nStops = tokenizer->eosTokenIds.size();
    char** s = new char*[nStops];
    for (unsigned int i = 0; i < nStops; i++) {
        s[i] = tokenizer->vocab[tokenizer->eosTokenIds[i]];
    }

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

static const char *chatTemplateTypeToString(const ChatTemplateType type) {
    if (type == TEMPLATE_LLAMA2) return "llama2";
    if (type == TEMPLATE_LLAMA3) return "llama3";
    if (type == TEMPLATE_DEEP_SEEK3) return "deepSeek3";
    if (type == TEMPLATE_CHATML) return "chatml";
    return "unknown";
}

ChatTemplateGenerator::ChatTemplateGenerator(const ChatTemplateType type, const char* chatTemplate, const char* eos)
    : buffer() 
{
    if (type == TEMPLATE_UNKNOWN) {
        if (chatTemplate == NULL)
            throw std::runtime_error("The tokenizer does not include chat template");
        if (strstr(chatTemplate, "[INST]") != NULL) {
            this->type = TEMPLATE_LLAMA2;
        } else if (strstr(chatTemplate, "<|start_header_id|>") != NULL) {
            this->type = TEMPLATE_LLAMA3;
        } else if (strstr(chatTemplate, "<｜Assistant｜>") != NULL) {
            this->type = TEMPLATE_DEEP_SEEK3;
        } else if (strstr(chatTemplate, "<|im_start|>") != NULL) {
            this->type = TEMPLATE_CHATML;
        } else {
            throw std::runtime_error("Not supported chat template");
        }
    } else {
        this->type = type;
    }
    this->eos = eos;

    printf("⭐ Chat template: %s\n", chatTemplateTypeToString(this->type));
}

GeneratedChat ChatTemplateGenerator::generate(unsigned int nItems, ChatItem* items, bool appendGenerationPrompt) {
    buffer.clear();

    size_t publicPromptSize = 0;

    if (type == TEMPLATE_LLAMA2) {
        unsigned int i = 0;
        if (nItems >= 2 && items[0].role == "system" && items[1].role == "user") {
            buffer += "[INST] <<SYS>>\n" + items[0].message + "\n<</SYS>>\n\n" + items[1].message + " [/INST]" + eos;
            i += 2;
        }
        for (; i < nItems; i++) {
            if (items[i].role == "assistant") {
                buffer += items[i].message + eos;
            } else if (items[i].role == "user") {
                buffer += "[INST] " + items[i].message + " [/INST]" + eos;
            }
        }
    } else if (type == TEMPLATE_LLAMA3) {
        for (unsigned int i = 0; i < nItems; i++)
            buffer += "<|start_header_id|>" + items[i].role + "<|end_header_id|>\n\n" + items[i].message + eos;
        if (appendGenerationPrompt)
            buffer += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (type == TEMPLATE_DEEP_SEEK3) {
        unsigned int i = 0;
        if (nItems > 0 && items[0].role == "system") {
            buffer += items[0].message;
            i++;
        }
        for (; i < nItems; i++) {
            if (items[i].role == "user") {
                buffer += "<｜User｜>" + items[i].message;
            } else if (items[i].role == "assistant") {
                buffer += "<｜Assistant｜>" + items[i].message;
            }
        }
        if (appendGenerationPrompt) {
            buffer += "<｜Assistant｜><think>\n";
            publicPromptSize = 8; 
        }
    } else if (type == TEMPLATE_CHATML) {
        for (unsigned int i = 0; i < nItems; i++) {
            if (items[i].role == "system") {
                buffer += "<|im_start|>system\n" + items[i].message + "<|im_end|>\n";
            } else if (items[i].role == "user") {
                buffer += "<|im_start|>user\n" + items[i].message + "<|im_end|>\n";
            } else if (items[i].role == "assistant") {
                buffer += "<|im_start|>assistant\n" + items[i].message + "<|im_end|>\n";
            }
            if (appendGenerationPrompt)
                buffer += "<|im_start|>assistant\n";
        }
    }

    const char *content = buffer.c_str();
    size_t length = buffer.size();
    const char *publicPrompt = publicPromptSize > 0
        ? &content[length - publicPromptSize]
        : nullptr;
#if DEBUG_TEMPLATE_GENERATOR
    printf("\033[1;31m[%s]\033[0m", content);
#endif
    return {content, length, publicPrompt};
}

EosDetector::EosDetector(size_t nTokens, const int *tokens, const char** pieces, int paddingLeft, int paddingRight) {
    this->nTokens = nTokens;
    this->tokens = tokens;
    this->pieces = pieces;
    this->pieceSizes = new size_t[nTokens];
    for (size_t s = 0; s < nTokens; s++) {
        pieceSizes[s] = strlen(pieces[s]);
        printf("🛑 Stop: %s\n", pieces[s]);
    }
    this->bufferPos = 0;
    this->bufferSize = 0;
    this->paddingLeft = paddingLeft;
    this->paddingRight = paddingRight;
}

EosDetector::~EosDetector() {
    if (bufferSize > 0)
        delete[] buffer;
    delete[] pieceSizes;
}

bool EosDetector::isEos(int tokenId) {
    for (size_t i = 0; i < nTokens; i++) {
        if (tokenId == tokens[i])
            return true;
    }
    return false;
}

EosDetectorType EosDetector::append(int tokenId, const char *piece) {
    if (piece != nullptr) {
        int pieceLength = std::strlen(piece);
        int newSize = bufferPos + pieceLength + 1;
        if (newSize > bufferSize) {
            char* newBuffer = new char[newSize];
            if (bufferPos > 0)
                std::memcpy(newBuffer, buffer, bufferPos);
            if (bufferSize > 0)
                delete[] buffer;
            buffer = newBuffer;
            bufferSize = newSize;
        }
        std::memcpy(&buffer[bufferPos], piece, pieceLength);
        bufferPos += pieceLength;
        buffer[bufferPos] = '\0';
    }

    // detection

    if (isEos(tokenId)) {
        eosPos = bufferPos;
        return EOS;
    }
    eosPos = -1;

    for (size_t s = 0; s < nTokens; s++) {
        size_t pieceSize = pieceSizes[s];
        if (bufferPos > pieceSize + paddingLeft + paddingRight) continue;

        for (int lo = 0; lo <= paddingLeft; lo++) {
            int n = bufferPos - lo;
            if (n == 0 || n > pieceSize + paddingRight) continue;
            if (n > pieceSize) n = pieceSize;
            if (strncmp(buffer + lo, pieces[s], n) == 0) {
                if (n == pieceSize) {
                    eosPos = lo;
                    buffer[eosPos] = '\0';
                    return EOS;
                }
                return MAYBE_EOS;
            }
        }
    }
    return NOT_EOS;
}

char* EosDetector::getDelta() {
    if (bufferPos == 0) return nullptr;
    if (eosPos == -1) return buffer;
    if (eosPos == 0) return nullptr;
    return buffer;
}

void EosDetector::reset() {
    bufferPos = 0;
}
