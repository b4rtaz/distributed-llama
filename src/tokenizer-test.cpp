#include <cassert>
#include <cstring>
#include "tokenizer.hpp"

#define DEV_TESTS false

#define ASSERT_EQ(a, b) \
    if (a != b) { \
        printf("Assertion failed: %d != %d (%s:%d)\n", a, b, __FILE__, __LINE__); \
        exit(-1); \
    }

#define TEST_EOS_ID 10000

void printOk(const char *name) {
    printf("✅ %24s passed\n", name);
}

void compare(const char *name, const int *a, const int *b, const unsigned int aN, const int bN) {
    bool passed = true;
    if (aN != bN) {
        passed = false;
    } else {
        for (unsigned int i = 0; i < bN; i++) {
            if (a[i] != b[i]) {
                passed = false;
                break;
            }
        }
    }
    if (!passed) {
        printf("❌ %24s failed\na: ", name);
        for (unsigned int j = 0; j < aN; j++)
            printf("%5d ", a[j]);
        printf("\nb: ");
        for (unsigned int j = 0; j < bN; j++)
            printf("%5d ", b[j]);
        printf("\n");
        exit(1);
    }
    printOk(name);
}

void dev_testEncode(Tokenizer *tokenizer) {
    int tokens[1024];
    int nTokens;

    {
        const char *text = "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        const int expectedTokens[] = {128000, 128006, 882, 128007, 271, 15339, 128009, 128006, 78191, 128007, 271};

        tokenizer->encode((char *)text, tokens, &nTokens, true, true);
        compare("case0", expectedTokens, tokens, 11, nTokens);
    }
    {
        const char *text = "!!&&@(*x)^^!";
        const int expectedTokens[] = {128000, 3001, 7827, 31, 4163, 87, 8, 22634, 0};

        tokenizer->encode((char *)text, tokens, &nTokens, true, true);
        compare("case1", expectedTokens, tokens, 9, nTokens);
    }
    {
        const char *text = "😃!😇x";
        const int expectedTokens[] = {128000, 76460, 225, 0, 76460, 229, 87};

        tokenizer->encode((char *)text, tokens, &nTokens, true, true);
        compare("case2", expectedTokens, tokens, 7, nTokens);
    }
}

void dev_testDecoderEmoji(Tokenizer *tokenizer) {
    char *x0 = tokenizer->decode(128000);
    assert(x0 == nullptr);

    char *x1 = tokenizer->decode(76460);
    assert(x1 == nullptr);

    char *x2 = tokenizer->decode(225);
    assert(x2 == nullptr);

    char *x3 = tokenizer->decode(0);
    assert(strstr(x3, "😃!") != NULL);

    char *x4 = tokenizer->decode(56);
    assert(strstr(x3, "Y") != NULL);

    printOk("testDecoderEmoji");
}

void dev_testDecoderEmojiWithEos(Tokenizer *tokenizer) {
    char *x0 = tokenizer->decode(128000);
    char *x1 = tokenizer->decode(76460);
    char *x2 = tokenizer->decode(225);
    char *x3 = tokenizer->decode(128001);

    assert(x0 == nullptr);
    assert(x1 == nullptr);
    assert(x2 == nullptr);
    assert(strstr(x3, "😃") != NULL); // piece should not contain <|end_of_text|>
    printOk("decoderEmojiWithEos");
}

void testChatTemplateDetection() {
    ChatTemplateGenerator t0(TEMPLATE_UNKNOWN, "{\% set loop_messages = messages \%}{\% for message in loop_messages \%}{\% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' \%}{\% if loop.index0 == 0 \%}{\% set content = bos_token + content \%}{\% endif \%}{{ content }}{\% endfor \%}{\% if add_generation_prompt \%}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{\% endif \%}", "<eos>");
    assert(t0.type == TEMPLATE_LLAMA3);

    printOk("chatTemplateDetection");
}

void testEosDetectorWithPadding() {
    const int tokens[2] = {TEST_EOS_ID, TEST_EOS_ID + 1};
    const char *pieces[2] = { "<eos>", "<stop>" };
    EosDetector detector(2, tokens, pieces, 1, 1);

    // "<eos>"
    {
        ASSERT_EQ(detector.append(1, "<"), MAYBE_EOS);
        ASSERT_EQ(detector.append(2, "eo"), MAYBE_EOS);
        ASSERT_EQ(detector.append(3, "s>"), EOS);
        assert(detector.getDelta() == nullptr);
    }

    // "<stop> "
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "<"), MAYBE_EOS);
        ASSERT_EQ(detector.append(2, "stop"), MAYBE_EOS);
        ASSERT_EQ(detector.append(3, "> "), EOS);
        assert(detector.getDelta() == nullptr);
    }

    // " "
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, " "), NOT_EOS);

        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, " ") == 0);
    }

    // "!<eos> "
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "!<"), MAYBE_EOS);
        ASSERT_EQ(detector.append(2, "eos"), MAYBE_EOS);
        ASSERT_EQ(detector.append(3, "> "), EOS);

        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "!") == 0);
    }

    // "!<eos> "
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "<eo"), MAYBE_EOS);
        ASSERT_EQ(detector.append(2, "s>XY"), NOT_EOS);

        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "<eos>XY") == 0);
    }

    // "<eo" + EOS
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "<eo"), MAYBE_EOS);
        ASSERT_EQ(detector.append(TEST_EOS_ID, nullptr), EOS);

        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "<eo") == 0);
    }

    // EOS
    detector.reset();
    {
        ASSERT_EQ(detector.append(TEST_EOS_ID, nullptr), EOS);
        assert(detector.getDelta() == nullptr);
    }

    // after reset it's expected to return nullptr delta if to the append() passed nullptr piece
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "x"), NOT_EOS);
        char *delta0 = detector.getDelta();
        assert(std::strcmp(delta0, "x") == 0);

        detector.reset();

        ASSERT_EQ(detector.append(2, nullptr), NOT_EOS);
        char *delta1 = detector.getDelta();
        assert(delta1 == nullptr);
    }

    printOk("eosDetectorWithPadding");
}

void testEosDetectorWithLongPadding() {
    const int tokens[1] = {TEST_EOS_ID};
    const char *pieces[1] = { "|end|" };
    EosDetector detector(1, tokens, pieces, 5, 5);

    // "lipsum"
    {
        ASSERT_EQ(detector.append(1, "lipsum"), NOT_EOS);
        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "lipsum") == 0);
    }

    // "lorem"
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "lorem"), NOT_EOS);
        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "lorem") == 0);
    }

    // "lorem|enQ"
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "lorem|"), MAYBE_EOS);
        ASSERT_EQ(detector.append(2, "enQ"), NOT_EOS);
        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "lorem|enQ") == 0);
    }

    printOk("eosDetectorWithLongPadding");
}

void testEosDetectorWithoutPadding() {
    const int tokens[1] = {TEST_EOS_ID};
    const char *pieces[1] = { "<eos>" };
    EosDetector detector(1, tokens, pieces, 0, 0);

    // "<eos>"
    {
        ASSERT_EQ(detector.append(1, "<"), MAYBE_EOS);
        ASSERT_EQ(detector.append(2, "eo"), MAYBE_EOS);
        ASSERT_EQ(detector.append(3, "s>"), EOS);
        assert(detector.getDelta() == nullptr);
    }

    // " <"
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, " <"), NOT_EOS);
        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, " <") == 0);
    }

    // "<eos> "
    detector.reset();
    {
        ASSERT_EQ(detector.append(1, "<eos"), MAYBE_EOS);
        ASSERT_EQ(detector.append(2, "> "), NOT_EOS);
        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "<eos> ") == 0);
    }

    // EOS
    detector.reset();
    {
        ASSERT_EQ(detector.append(TEST_EOS_ID, nullptr), EOS);
        assert(detector.getDelta() == nullptr);
    }

    // emoji
    detector.reset();
    {
        ASSERT_EQ(detector.append(TEST_EOS_ID, "😃"), EOS);
        char *delta = detector.getDelta();
        assert(delta != nullptr);
        assert(std::strcmp(delta, "😃") == 0);
    }

    printOk("eosDetectorWithLongPadding");
}

int main() {
#if DEV_TESTS
    Tokenizer tokenizer("models/llama3_2_1b_instruct_q40/dllama_tokenizer_llama3_2_1b_instruct_q40.t");
    dev_testEncode(&tokenizer);
    dev_testDecoderEmoji(&tokenizer);
    dev_testDecoderEmojiWithEos(&tokenizer);
#endif

    testChatTemplateDetection();
    testEosDetectorWithPadding();
    testEosDetectorWithLongPadding();
    testEosDetectorWithoutPadding();
    return 0;
}
