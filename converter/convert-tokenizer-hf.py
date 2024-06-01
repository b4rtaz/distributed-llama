import sys
import json
import os
from sentencepiece import SentencePieceProcessor
writer = __import__('tokenizer-writer')

def openJson(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

class TokensResolver:
    def __init__(self, dirPath, tokenizerConfig):
        self.dirPath = dirPath
        self.tokenizerConfig = tokenizerConfig
        self.bosId = None
        self.eosId = None
        self.tokens = []
        self.scores = []

    def resolvePreTrainedTokenizerFast(self):
        tokenizer = openJson(os.path.join(self.dirPath, 'tokenizer.json'))
        assert(tokenizer['model']['type'] == 'BPE')

        i = 0
        for token in tokenizer['model']['vocab'].keys():
            assert(tokenizer['model']['vocab'][token] == i)
            self.tokens.append(token.encode('utf8'))
            self.scores.append(-float(i))
            i += 1
        if ('added_tokens' in tokenizer):
            for at in tokenizer['added_tokens']:
                assert(at['id'] == i)
                self.tokens.append(at['content'].encode('utf8'))
                self.scores.append(-float(i))
                if (at['content'] == self.tokenizerConfig['bos_token']):
                    self.bosId = i
                if (at['content'] == self.tokenizerConfig['eos_token']):
                    self.eosId = i
                i += 1

    def resolveLlamaTokenizer(self):
        modelPath = os.path.join(self.dirPath, 'tokenizer.model')
        processor = SentencePieceProcessor(model_file=modelPath)

        assert processor.vocab_size() == processor.get_piece_size()
        self.bosId = processor.bos_id()
        self.eosId = processor.eos_id()

        vocabSize = processor.vocab_size()
        for i in range(vocabSize):
            t = processor.id_to_piece(i)
            s = processor.get_score(i)
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8')
            self.tokens.append(b)
            self.scores.append(s)

    def resolve(self):
        cls = self.tokenizerConfig['tokenizer_class']
        if (cls == 'PreTrainedTokenizerFast'):
            return self.resolvePreTrainedTokenizerFast()
        if (cls == 'LlamaTokenizer'):
            return self.resolveLlamaTokenizer()
        raise Exception(f'Tokenizer {cls} is not supported')

def printUsage():
    print('Usage: python convert-tokenizer-hf.py <tokenizerFolderPath> <name>')
    print()
    print('Options:')
    print('  <tokenizerFolderPath> The path to the folder with tokenizer_config.json')
    print('  <name>                The name of the tokenizer (e.g. "llama3")')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        printUsage()
        exit(1)

    dirPath = sys.argv[1]
    name = sys.argv[2]
    tokenizerConfig = openJson(os.path.join(dirPath, 'tokenizer_config.json'))

    resolver = TokensResolver(dirPath, tokenizerConfig)
    resolver.resolve()

    print(f'bosId: {resolver.bosId} ({resolver.tokens[resolver.bosId]})')
    print(f'eosId: {resolver.eosId} ({resolver.tokens[resolver.eosId]})')

    chatTemplate = None
    chatExtraStop = None
    if ('chat_template' in tokenizerConfig):
        chatTemplate = tokenizerConfig['chat_template'].encode('utf-8')
        input = input('⏩ Enter value for chat extra stop (enter to skip): ')
        if (input != ''):
            chatExtraStop = input.encode('utf-8')

    outputFileName = f'dllama_tokenizer_{name}.t'
    with open(outputFileName, 'wb') as outputFile:
        writer.writeTokenizer(outputFile, {
            'bos_id': resolver.bosId,
            'eos_id': resolver.eosId,
            'chat_eos_id': resolver.eosId,
        }, resolver.tokens, resolver.scores, chatTemplate, chatExtraStop)
    print(f'✅ Created {outputFileName}')
