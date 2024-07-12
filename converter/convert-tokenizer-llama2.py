import sys
import os
from sentencepiece import SentencePieceProcessor
writer = __import__('tokenizer-writer')

chatTemplate = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def printUsage():
    print('Usage: python convert-tokenizer-llama2.py <llama2FolderPath>')
    print()
    print('Options:')
    print('  <llama2FolderPath> The path to the folder with llama2 folder path')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        printUsage()
        exit(1)

    dirPath = sys.argv[1]
    modelPath = os.path.join(dirPath, 'tokenizer.model')
    processor = SentencePieceProcessor(model_file=modelPath)

    vocabSize = processor.vocab_size()
    tokens = []
    scores = []
    for i in range(vocabSize):
        t = processor.id_to_piece(i)
        s = processor.get_score(i)
        t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
        b = t.encode('utf-8')
        tokens.append(b)
        scores.append(s)

    outputFileName = 'dllama_tokenizer_llama2.t'
    with open(outputFileName, 'wb') as outputFile:
        writer.writeTokenizer(outputFile, {
            'bos_id': processor.bos_id(),
            'eos_id': processor.eos_id(),
            'chat_eos_id': processor.eos_id(),
        }, tokens, scores, chatTemplate.encode('utf-8'), None)

    print(f'✅ Created {outputFileName}')
