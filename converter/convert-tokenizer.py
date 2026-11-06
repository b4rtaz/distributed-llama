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
        self.regularVocabSize = None
        self.tokens = []
        self.scores = []

    def resolveConfiguredTokenIds(self, tokenizer, value):
        if value is None:
            return None
        values = value if isinstance(value, list) else [value]
        result = []
        for token in values:
            if isinstance(token, dict):
                token = token['content']
            tokenId = tokenizer.convert_tokens_to_ids(token)
            if tokenId is None:
                raise Exception(f'Cannot resolve configured token: {token}')
            result.append(tokenId)
        return result

    def resolvePreTrainedTokenizerFast(self):
        utb = unicodeToBytes()
        tokenizerData = openJson(os.path.join(self.dirPath, 'tokenizer.json'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_file = os.path.join(self.dirPath, 'tokenizer.json'))
        config = openJson(os.path.join(self.dirPath, 'config.json'))
        vocabLen = len(tokenizer.get_vocab())
        self.regularVocabSize = len(tokenizerData['model']['vocab'])
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

        # Pad tokenizer vocab to match model vocab_size if needed
        targetVocabSize = config.get('vocab_size', config.get('text_config', {}).get('vocab_size', vocabLen))
        if targetVocabSize > vocabLen:
            print(f'⚠️ Padding tokenizer vocab from {vocabLen} to {targetVocabSize}')
            for i in range(vocabLen, targetVocabSize):
                self.tokens.append(f'<|reserved_{i}|>'.encode('utf-8'))
                self.scores.append(-float(i))

        self.bosId = tokenizer.bos_token_id
        if self.bosId is None:
            bosIds = self.resolveConfiguredTokenIds(tokenizer, self.tokenizerConfig.get('bos_token'))
            if bosIds is not None:
                self.bosId = bosIds[0]
            else:
                self.bosId = config.get('bos_token_id', config.get('text_config', {}).get('bos_token_id'))

        if tokenizer.eos_token_id is not None:
            self.eosIds = [tokenizer.eos_token_id]
        else:
            self.eosIds = self.resolveConfiguredTokenIds(tokenizer, self.tokenizerConfig.get('eos_token'))

    def resolveLlamaTokenizer(self):
        modelPath = os.path.join(self.dirPath, 'tokenizer.model')
        processor = SentencePieceProcessor(model_file=modelPath)

        assert processor.vocab_size() == processor.get_piece_size()
        self.bosId = processor.bos_id()
        self.eosIds = [processor.eos_id()]
        vocabSize = processor.vocab_size()
        self.regularVocabSize = self.bosId
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
        if (cls == 'PreTrainedTokenizer' or
            cls == 'PreTrainedTokenizerFast' or
            cls == 'LlamaTokenizerFast' or
            cls == 'Qwen2Tokenizer'):
            self.resolvePreTrainedTokenizerFast()
        elif (cls == 'LlamaTokenizer'):
            self.resolveLlamaTokenizer()
        else:
            raise Exception(f'Tokenizer {cls} is not supported')
        self.resolveEosIds()

    def resolveEosIds(self):
        config = openJson(os.path.join(self.dirPath, 'config.json'))
        generationPath = os.path.join(self.dirPath, 'generation_config.json')
        generation = openJson(generationPath) if os.path.exists(generationPath) else {}
        if generation.get('eos_token_id') is not None:
            sources = [generation]
        else:
            sources = [config, config.get('text_config', {})]

        # Keep the tokenizer EOS first for chat templates.
        eosIds = list(self.eosIds or [])
        for source in sources:
            ids = source.get('eos_token_id')
            if ids is not None:
                eosIds.extend(ids if isinstance(ids, list) else [ids])
        self.eosIds = list(dict.fromkeys(eosIds))

def printUsage():
    print('Usage: python convert-tokenizer.py <tokenizerFolderPath> <name>')
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

    addBos = tokenizerConfig.get('add_bos_token', True)
    if addBos and resolver.bosId is None:
        raise Exception('Tokenizer requests BOS insertion but has no BOS token')
    if not resolver.eosIds:
        raise Exception('Cannot resolve eosIds')
    if resolver.bosId is None:
        print('bosId: none (disabled)')
    else:
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
            addBos,
            resolver.eosIds,
            resolver.regularVocabSize)
    print(f'✅ Created {outputFileName}')
