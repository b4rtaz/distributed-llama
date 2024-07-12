#include <cassert>
#include <cstring>
#include <cstdlib>
#include "tokenizer.hpp"

#define ASSERT_EOS_TYPE(type, expected) \
    if (type != expected) { \
        printf("Expected %d, got %d (line: %d)\n", expected, type, __LINE__); \
        exit(1); \
    }

#define EOS_ID 10000

void testChatTemplate() {
    ChatTemplate t0(TEMPLATE_UNKNOWN, "{\% set loop_messages = messages \%}{\% for message in loop_messages \%}{\% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' \%}{\% if loop.index0 == 0 \%}{\% set content = bos_token + content \%}{\% endif \%}{{ content }}{\% endfor \%}{\% if add_generation_prompt \%}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{\% endif \%}", "<eos>");
    assert(t0.type == TEMPLATE_LLAMA3);

    ChatTemplate t1(TEMPLATE_UNKNOWN, "{{bos_token}}{\% for message in messages \%}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{\% endfor \%}{\% if add_generation_prompt \%}{{ '<|im_start|>assistant\n' }}{\% endif \%}", "<eos>");
    assert(t1.type == TEMPLATE_CHATML);

    ChatTemplate t2(TEMPLATE_UNKNOWN, "{\% for message in messages \%}\n{\% if message['role'] == 'user' \%}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{\% elif message['role'] == 'system' \%}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{\% elif message['role'] == 'assistant' \%}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{\% endif \%}\n{\% if loop.last and add_generation_prompt \%}\n{{ '<|assistant|>' }}\n{\% endif \%}\n{\% endfor \%}", "<eos>");
    assert(t2.type == TEMPLATE_ZEPHYR);

    printf("✅ ChatTemplate\n");
}

void testEosDetectorWithPadding() {
    const char* stops[2] = { "<eos>", "<stop>" };
    EosDetector detector(EOS_ID, 2, stops, 1, 1);

    // "<eos>"
    {
        ASSERT_EOS_TYPE(detector.append(1, "<"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(2, "eo"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(3, "s>"), EOS);
        assert(detector.getDelta() == NULL);
    }

    // "<stop> "
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, "<"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(2, "stop"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(3, "> "), EOS);
        assert(detector.getDelta() == NULL);
    }

    // " "
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, " "), NOT_EOS);

        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, " ") == 0);
    }

    // "!<eos> "
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, "!<"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(2, "eos"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(3, "> "), EOS);

        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, "!") == 0);
    }

    // "!<eos> "
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, "<eo"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(2, "s>XY"), NOT_EOS);

        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, "<eos>XY") == 0);
    }

    // "<eo" + EOS
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, "<eo"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(EOS_ID, "<eos>"), EOS);

        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, "<eo") == 0);
    }

    // EOS
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(EOS_ID, "<eos>"), EOS);
        assert(detector.getDelta() == NULL);
    }

    printf("✅ EosDetector with padding\n");
}


void testEosDetectorWithLongPadding() {
    const char* stops[1] = { "|end|" };
    EosDetector detector(EOS_ID, 1, stops, 5, 5);

    // "lipsum"
    {
        ASSERT_EOS_TYPE(detector.append(1, "lipsum"), NOT_EOS);
        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, "lipsum") == 0);
    }

    // "lorem"
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, "lorem"), NOT_EOS);
        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, "lorem") == 0);
    }

    // "lorem|enQ"
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, "lorem|"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(2, "enQ"), NOT_EOS);
        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, "lorem|enQ") == 0);
    }

    printf("✅ EosDetector with long padding\n");
}

void testEosDetectorWithoutPadding() {
    const char* stops[1] = { "<eos>" };
    EosDetector detector(EOS_ID, 1, stops, 0, 0);

    // "<eos>"
    {
        ASSERT_EOS_TYPE(detector.append(1, "<"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(2, "eo"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(3, "s>"), EOS);
        assert(detector.getDelta() == NULL);
    }

    // " <"
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, " <"), NOT_EOS);
        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, " <") == 0);
    }

    // "<eos> "
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(1, "<eos"), MAYBE_EOS);
        ASSERT_EOS_TYPE(detector.append(2, "> "), NOT_EOS);
        char* delta = detector.getDelta();
        assert(delta != NULL);
        assert(strcmp(delta, "<eos> ") == 0);
    }

    // EOS
    detector.clear();
    {
        ASSERT_EOS_TYPE(detector.append(EOS_ID, "<eos>"), EOS);
        assert(detector.getDelta() == NULL);
    }

    printf("✅ EosDetector without padding\n");
}

int main() {
    testChatTemplate();
    testEosDetectorWithPadding();
    testEosDetectorWithLongPadding();
    testEosDetectorWithoutPadding();
    return EXIT_SUCCESS;
}
