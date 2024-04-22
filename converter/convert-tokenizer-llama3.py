import sys
import struct
import base64

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

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Invalid usage')
        exit(1)

    modelPath = sys.argv[1]

    with open(modelPath, 'r') as inputFile:
        with open('dllama-llama3-tokenizer.t', 'wb') as outputFile:
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

            vocabSize = len(tokens)
            maxTokenLength = max(len(t) for t in tokens)

            outputFile.write(struct.pack('IIIiii',
                0x567123,
                vocabSize,
                maxTokenLength,
                bosId,
                eosId,
                -1))

            for i in range(0, vocabSize):
                outputFile.write(struct.pack('fI', scores[i], len(tokens[i])))
                outputFile.write(tokens[i])

            print(f'maxTokenLength={maxTokenLength}')
            print(f'bosId={bosId}')
            print(f'eosId={eosId}')
            print(f'vocabSize={vocabSize}')