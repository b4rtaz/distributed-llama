import struct

def writeTokenizer(file, tokens, scores, chatTemplate, bosId, addBos, eosTokens, regularVocabSize):
    headerKeys = {
        'version': 0,
        'vocab_size': 1,
        'max_token_length': 2,
        'bos_id': 3,
        'chat_template': 7,
        'n_eos_tokens': 9,
        'add_bos': 10,
        'regular_vocab_size': 11,
    }
    header = struct.pack('i', 0x567124)

    nTokens = len(tokens)
    maxTokenLength = max(len(t) for t in tokens)
    if regularVocabSize is None:
        raise Exception('regularVocabSize is required')
    if regularVocabSize < 0 or regularVocabSize > nTokens:
        raise Exception(f'Invalid regularVocabSize: {regularVocabSize}')
    if addBos and bosId is None:
        raise Exception('addBos requires bosId')

    params = {}
    params['bos_id'] = bosId
    params['version'] = 1
    params['vocab_size'] = nTokens
    params['max_token_length'] = maxTokenLength
    if (chatTemplate):
        params['chat_template'] = len(chatTemplate)
    params['n_eos_tokens'] = len(eosTokens)
    params['add_bos'] = 1 if addBos else 0
    params['regular_vocab_size'] = regularVocabSize

    data = b''
    for key in params:
        value = params[key]
        if value is None:
            continue
        if key in headerKeys:
            data += struct.pack('ii', headerKeys[key], params[key])
        else:
            print(f'Unknown header key: {key}')

    print('⭐ Params:')
    print(params)
    if (chatTemplate):
        print('⭐ Chat template:')
        print(chatTemplate)

    header += struct.pack('i', len(header) * 2 + len(data))
    file.write(header)
    file.write(data)
    if chatTemplate:
        file.write(chatTemplate)

    for eosToken in eosTokens:
        file.write(struct.pack('i', eosToken))

    for i in range(0, nTokens):
        size = len(tokens[i])
        assert(size > 0)
        file.write(struct.pack('fI', scores[i], size))
        file.write(tokens[i])
