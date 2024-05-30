import struct

def writeTokenizer(file, params, chatTemplate, tokens, scores):
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
    params['version'] = 0
    params['vocab_size'] = nTokens
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
        chatTemplateValue = list(chatTemplate.values())
        nChatTemplates = len(chatTemplateValue)
        for i in range(0, nChatTemplates):
            file.write(struct.pack('I', len(chatTemplateValue[i])))
        for i in range(0, nChatTemplates):
            file.write(chatTemplateValue[i].encode())

    for i in range(0, nTokens):
        file.write(struct.pack('fI', scores[i], len(tokens[i])))
        file.write(tokens[i])
