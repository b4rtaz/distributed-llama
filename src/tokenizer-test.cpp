#include "tokenizer.hpp"

int main() {
    Tokenizer tokenizer("models/llama3_2_1b_instruct_q40/dllama_tokenizer_llama3_2_1b_instruct_q40.t");

    int tokens[1024];
    int nTokens;

    tokenizer.encode((char *)"<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        tokens, &nTokens, true, true);

    // expected: 128000, 128006, 882, 128007, 271, 15339, 128009, 128006, 78191, 128007, 271

    for (int i = 0; i < nTokens; i++)
        printf("%d, ", tokens[i]);
    return 0;
}
