import struct

def writeTokenizer(file, params, chatTemplate, tokens, scores):
    assert(params['eos_id'] is not None)
    assert(params['bos_id'] is not None)

    headerKeys = {
        'version': 0,
        'vocab_size': 1,
        'max_token_length': 2,
        'bos_id': 3,
        'eos_id': 4,
        'pad_id': 5,
        'chat_eos_id': 6,
        'chat_template': 7
    }
    header = struct.pack('i', 0x567124)

    nTokens = len(tokens)
    maxTokenLength = max(len(t) for t in tokens)

    params['version'] = 0
    params['vocab_size'] = nTokens
    params['max_token_length'] = maxTokenLength
    if (chatTemplate):
        params['chat_template'] = len(chatTemplate)

    data = b''
    for key in params:
        if key in headerKeys:
            data += struct.pack('ii', headerKeys[key], params[key])
        else:
            print(f'Unknown header key: {key}')

    header += struct.pack('i', len(header) * 2 + len(data))
    file.write(header)
    file.write(data)

    print(params)
    if (chatTemplate):
        print(chatTemplate)

    if (chatTemplate):
        chatTemplateValue = list(chatTemplate.values())
        nChatTemplates = len(chatTemplateValue)
        for i in range(0, nChatTemplates):
            file.write(struct.pack('I', len(chatTemplateValue[i].encode('utf8'))))
        for i in range(0, nChatTemplates):
            data = chatTemplateValue[i].encode('utf8')
            if (len(data) > 0):
                file.write(data)

    for i in range(0, nTokens):
        size = len(tokens[i])
        assert(size > 0)
        file.write(struct.pack('fI', scores[i], size))
        file.write(tokens[i])
