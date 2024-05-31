import sys
import json
import os
writer = __import__('tokenizer-writer')

def openJson(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def printUsage():
    print('Usage: python convert-tokenizer-hf.py <tokenizerFolderPath> <name>')
    print()
    print('Options:')
    print('  <tokenizerFolderPath> The path to the folder with tokenizer.json and tokenizer_config.json')
    print('  <name>                The name of the tokenizer (e.g. "llama3")')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        printUsage()
        exit(1)
    
    dirPath = sys.argv[1]
    name = sys.argv[2]
    tokenizerConfig = openJson(os.path.join(dirPath, 'tokenizer_config.json'))
    tokenizer = openJson(os.path.join(dirPath, 'tokenizer.json'))

    assert(tokenizerConfig['tokenizer_class'] == 'PreTrainedTokenizerFast')
    assert(tokenizer['model']['type'] == 'BPE')
    i = 0
    tokens = []
    scores = []
    bosId = None
    eosId = None
    for token in tokenizer['model']['vocab'].keys():
        assert(tokenizer['model']['vocab'][token] == i)
        tokens.append(token.encode('utf8'))
        scores.append(-float(i))
        i += 1
    if ('added_tokens' in tokenizer):
        for at in tokenizer['added_tokens']:
            assert(at['id'] == i)
            tokens.append(at['content'].encode('utf8'))
            scores.append(-float(i))
            if (at['content'] == tokenizerConfig['bos_token']):
                bosId = i
            if (at['content'] == tokenizerConfig['eos_token']):
                eosId = i
            i += 1

    templateChat = None
    if ('chat_template' in tokenizerConfig):
        template = tokenizerConfig['chat_template']
        print('⭐ Found chat template:')
        print()
        print(template.replace('\n', '\\n'))
        print()
        print('⭐ To create the tokenizer file you need to manually specify chat template values. Enter \\n for new line.')
        templateChat = {}
        templateKeys = ['chat_message_start', 'chat_role_start', 'chat_role_end', 'chat_message_end', 'chat_generation_prompt', 'chat_extra_stop']
        for key in templateKeys:
            value = input(f'⏩ Enter value for chat template key "{key}":\n')
            templateChat[key] = value.replace('\\n', '\n')

    outputFileName = f'dllama_tokenizer_{name}.t'
    with open(outputFileName, 'wb') as outputFile:
        writer.writeTokenizer(outputFile, {
            'bos_id': bosId,
            'eos_id': eosId,
            'chat_eos_id': eosId,
        }, templateChat, tokens, scores)
    print(f'✅ Created {outputFileName}')
