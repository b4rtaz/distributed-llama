#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>
#include "funcs.hpp"
#include "shared-buffer.hpp"
#include "transformer.hpp"
#include "tokenizer.hpp"

void generate2(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char empty_prompt[] = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    tokenizer->encode(prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {
        transformer->forward(token, pos);
        float* logits = transformer->logits;

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler->sample(logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = tokenizer->decode(token, next);
        safePrintf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = timeMs(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = timeMs();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void generate(Transformer* transformer) {
    float temperature = 0.8f;
    float topp = 0.9f; // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    unsigned long long rng_seed = (unsigned int)time(NULL);
    char* tokenizer_path = (char*)"/Users/b4rtaz/Dev/llama2.c/tokenizer.bin";
    //int steps = spec->seqLen;
    int steps = 256;

    Tokenizer tokenizer(tokenizer_path, transformer->spec->vocabSize);
    Sampler sampler(transformer->spec->vocabSize, temperature, topp, rng_seed);

    char prompt[] = "Hello";
    generate2(transformer, &tokenizer, &sampler, prompt, steps);
}

int main() {
    initQuants();

    // const char* path = "./converter/llama_7b_fp32.bin"; FloatType type = F32;
    // const char* path = "./converter/llama_7b_torch.float16.bin"; FloatType type = F16;
    const char* path = "./converter/llama_7b_q40.bin"; FloatType type = Q40;
    const int sliceCount = 8;

    TransformerSpec* spec;
    Transformer* transformer;
    loadTransformer(&spec, &transformer, path, type, sliceCount);

    generate(transformer);

    delete transformer;
    delete spec;
    return EXIT_SUCCESS;
}
