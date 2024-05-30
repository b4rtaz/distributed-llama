import sys
import struct
import base64
writer = __import__('tokenizer-writer')

# Format of input file:
# ```
# IQ== 0
# Ig== 1
# Iw== 2
# ...
# ```

nSpecialTokens = 256
specialTokens = [
    '<|begin_of_text|>',
    '<|end_of_text|>',
    '<|reserved_special_token_0|>',
    '<|reserved_special_token_1|>',
    '<|reserved_special_token_2|>',
    '<|reserved_special_token_3|>',
    '<|start_header_id|>',
    '<|end_header_id|>',
    '<|reserved_special_token_4|>',
    '<|eot_id|>',
] + [
    f'<|reserved_special_token_{i}|>'
    for i in range(5, nSpecialTokens - 5)
]
bosId = 128000
eosId = 128001
chatEosId = 128009
chatTemplate = {
    'chat_message_start': '',
    'chat_role_start': '<|start_header_id|>',
    'chat_role_end': '<|end_header_id|>\n\n',
    'chat_message_end': '<|eot_id|>',
    'chat_generation_prompt': '<|start_header_id|>assistant<|end_header_id|>\n\n',
}

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Invalid usage')
        exit(1)

    modelPath = sys.argv[1]
    outputFileName = 'dllama_tokenizer_llama3.t'

    with open(modelPath, 'r') as inputFile:
        with open(outputFileName, 'wb') as outputFile:
            inputLines = inputFile.readlines()
            nLines = len(inputLines)

            tokens = []
            scores = []
            for line in inputLines:
                s = line.split(' ')
                bytes = base64.b64decode(s[0])
                score = -float(s[1])
                tokens.append(bytes)
                scores.append(score)

            specialTokenIndex = nLines
            for token in specialTokens:
                bytes = token.encode('utf-8')
                score = -float(specialTokenIndex)
                tokens.append(bytes)
                scores.append(score)
                specialTokenIndex += 1

            maxTokenLength = max(len(t) for t in tokens)

            writer.writeTokenizer(outputFile, {
                'max_token_length': maxTokenLength,
                'bos_id': bosId,
                'eos_id': eosId,
                'chat_eos_id': chatEosId,
            }, chatTemplate, tokens, scores)

    print(f'✅ Created {outputFileName}')
