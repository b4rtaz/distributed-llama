import sys
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
chatTemplate = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

def printUsage():
    print('Usage: python convert-tokenizer-llama3.py <tokenizerPath>')
    print()
    print('Options:')
    print('  <tokenizerPath> The path to the Llama 3 tokenizer model (tokenizer.model)')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        printUsage()
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

            writer.writeTokenizer(outputFile, {
                'bos_id': bosId,
                'eos_id': eosId,
                'chat_eos_id': chatEosId,
            }, tokens, scores, chatTemplate.encode('utf-8'), None)

    print(f'✅ Created {outputFileName}')
