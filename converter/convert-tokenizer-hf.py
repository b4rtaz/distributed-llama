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
        tokenConfig = openJson(os.path.join(self.dirPath, 'config.json'))
        tokenizerConfig = self.tokenizerConfig

        # Register special tokens from tokenizer_config.json if not already set
        specialTokens = {}
        for st in ['eos_token', 'pad_token', 'unk_token', 'bos_token']:
            if st in tokenizerConfig and tokenizerConfig[st] is not None:
                specialTokens[st] = tokenizerConfig[st]
        if specialTokens:
            tokenizer.add_special_tokens(specialTokens)

        # Enumerate main vocab + added tokens
        tokenizerJson = openJson(os.path.join(self.dirPath, 'tokenizer.json'))
        addedTokens = tokenizerJson.get('added_tokens', [])
        maxAddedId = max((t['id'] for t in addedTokens if 'id' in t), default=-1)
        vocabLen = max(len(tokenizer.get_vocab()), maxAddedId + 1)

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

        # Determine target vocab size from model config (top-level or text_config)
        targetVocabSize = tokenConfig.get('vocab_size')
        if targetVocabSize is None:
            tc = tokenConfig.get('text_config', {})
            targetVocabSize = tc.get('vocab_size', vocabLen)

        if targetVocabSize > vocabLen:
            print(f'⚠️ Padding tokenizer vocab from {vocabLen} to {targetVocabSize}')
            for i in range(vocabLen, targetVocabSize):
                self.tokens.append(f'<|reserved_{i}|>'.encode('utf-8'))
                self.scores.append(-float(i))

        # Resolve BOS: tokenizer → tokenizer_config → model config → text_config
        self.bosId = tokenizer.bos_token_id
        if self.bosId is None:
            self.bosId = tokenizerConfig.get('bos_token_id')
        if self.bosId is None:
            self.bosId = tokenConfig.get('bos_token_id')
        if self.bosId is None:
            tc = tokenConfig.get('text_config', {})
            self.bosId = tc.get('bos_token_id')

        # Resolve EOS: tokenizer → model config (eos_token_id) → tokenizer_config string
        if tokenizer.eos_token_id is not None:
            self.eosIds = [tokenizer.eos_token_id]
        else:
            eos = tokenConfig.get('eos_token_id')
            if eos is None:
                tc = tokenConfig.get('text_config', {})
                eos = tc.get('eos_token_id')
            if eos is None:
                eosStr = tokenizerConfig.get('eos_token')
                if eosStr is not None:
                    eos = tokenizer.convert_tokens_to_ids(eosStr)
            if isinstance(eos, list):
                self.eosIds = eos
            elif eos is not None:
                self.eosIds = [eos]

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
        if (cls == 'PreTrainedTokenizer' or
            cls == 'PreTrainedTokenizerFast' or
            cls == 'LlamaTokenizerFast' or
            cls == 'Qwen2Tokenizer'):
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

    if (resolver.eosIds is None):
        raise Exception('Cannot resolve eosIds')
    if (resolver.bosId is None):
        print('⚠️ No BOS token found, disabling add_bos')
    else:
        print(f'bosId: {resolver.bosId} ({resolver.tokens[resolver.bosId]})')
    for eosId in resolver.eosIds:
        print(f'eosId: {eosId} ({resolver.tokens[eosId]})')

    chatTemplate = None
    if ('chat_template' in tokenizerConfig):
        chatTemplate = tokenizerConfig['chat_template'].encode('utf-8')
    if chatTemplate is None:
        jinjaPath = os.path.join(dirPath, 'chat_template.jinja')
        if os.path.exists(jinjaPath):
            with open(jinjaPath, 'r', encoding='utf-8') as f:
                chatTemplate = f.read().encode('utf-8')

    addBos = True
    if ('add_bos_token' in tokenizerConfig):
        addBos = tokenizerConfig['add_bos_token']
    if (resolver.bosId is None):
        addBos = False

    outputFileName = f'dllama_tokenizer_{name}.t'
    with open(outputFileName, 'wb') as outputFile:
        writer.writeTokenizer(
            outputFile,
            resolver.tokens,
            resolver.scores,
            chatTemplate,
            resolver.bosId,
            addBos,
            resolver.eosIds)
    print(f'✅ Created {outputFileName}')
