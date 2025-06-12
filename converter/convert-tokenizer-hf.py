import sys
import json
import os
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizerFast
writer = __import__('tokenizer-writer')

def openJson(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def unicodeToBytes():
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))

class TokensResolver:
    def __init__(self, dirPath, tokenizerConfig):
        self.dirPath = dirPath
        self.tokenizerConfig = tokenizerConfig
        self.bosId = None
        self.eosIds = None
        self.tokens = []
        self.scores = []

    def resolvePreTrainedTokenizerFast(self):
        utb = unicodeToBytes()
        tokenizer = PreTrainedTokenizerFast(tokenizer_file = os.path.join(self.dirPath, 'tokenizer.json'))
        vocabLen = len(tokenizer.get_vocab())
        for i in range(vocabLen):
            tokenChars = list(tokenizer.convert_ids_to_tokens([i])[0])
            tokenBytes = []
            for chr in tokenChars:
                if (chr in utb):
                    tokenBytes.append(utb[chr])
                else:
                    tokenBytes += list(chr.encode('utf-8'))
            self.tokens.append(bytes(tokenBytes))
            self.scores.append(-float(i))

        self.bosId = tokenizer.bos_token_id
        if (tokenizer.eos_token_id):
            self.eosIds = [tokenizer.eos_token_id]
        if (self.bosId is None or self.eosId is None):
            config = openJson(os.path.join(self.dirPath, 'config.json'))
            if (self.bosId is None):
                self.bosId = config['bos_token_id']
            if (self.eosIds is None):
                self.eosIds = config['eos_token_id']
                if isinstance(self.eosIds, list):
                    self.eosIds = self.eosIds
                else:
                    self.eosIds = [self.eosIds]

    def resolveLlamaTokenizer(self):
        modelPath = os.path.join(self.dirPath, 'tokenizer.model')
        processor = SentencePieceProcessor(model_file=modelPath)

        assert processor.vocab_size() == processor.get_piece_size()
        self.bosId = processor.bos_id()
        self.eosIds = [processor.eos_id()]
        vocabSize = processor.vocab_size()
        for i in range(vocabSize):
            t = processor.id_to_piece(i)
            s = processor.get_score(i)
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            # Check for byte characters
            if len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
                # For example, "<0x0A>"" is a newline character
                b = bytearray.fromhex(t[3:-1])
            else:
                b = t.encode('utf-8')
            self.tokens.append(b)
            self.scores.append(s)

    def resolve(self):
        cls = self.tokenizerConfig['tokenizer_class']
        if (cls == 'PreTrainedTokenizerFast' or cls == 'LlamaTokenizerFast'):
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

    if (resolver.bosId is None or resolver.eosIds is None):
        raise Exception('Cannot resolve bosId or eosIds')
    print(f'bosId: {resolver.bosId} ({resolver.tokens[resolver.bosId]})')
    for eosId in resolver.eosIds:
        print(f'eosId: {eosId} ({resolver.tokens[eosId]})')

    chatTemplate = None
    if ('chat_template' in tokenizerConfig):
        chatTemplate = tokenizerConfig['chat_template'].encode('utf-8')

    outputFileName = f'dllama_tokenizer_{name}.t'
    with open(outputFileName, 'wb') as outputFile:
        writer.writeTokenizer(
            outputFile,
            resolver.tokens,
            resolver.scores,
            chatTemplate,
            resolver.bosId,
            resolver.eosIds)
    print(f'✅ Created {outputFileName}')
