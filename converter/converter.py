import os
import sys
import struct
import json
import torch
import numpy as np
from pathlib import Path
from torch import nn

def quantizeQ40(x, file):
    groupSize = 32
    groupSizeHalf = groupSize // 2
    assert(x.shape[0] % groupSize == 0)
    groups = x.reshape(-1, groupSize)
    gmax = np.max(groups, axis=1)
    gmin = np.min(groups, axis=1)
    absMax = np.where(-gmin > gmax, gmin, gmax)

    bytes = 0
    for gi, group in enumerate(groups):
        groupAbsMax = absMax[gi]
        delta = groupAbsMax / -8
        id = (1.0 / delta) if delta else 0

        twoBytes = struct.pack('e', np.float16(delta).astype(np.float16))
        file.write(twoBytes)
        bytes += 2

        for i in range(0, groupSizeHalf):
            x0 = group[i] * id + 8.5
            x1 = group[i + groupSizeHalf] * id + 8.5
            xi0 = min(15, int(x0))
            xi1 = min(15, int(x1))
            b = (xi0 & 0xF) | ((xi1 & 0xF) << 4)
            byte = struct.pack('B', b)
            file.write(byte)
            bytes += 1
    print(f'Quantized {x.shape[0] * 4} bytes into {bytes} bytes')

def exportTensor(file, tensor, floatType):
    tensor = nn.Parameter(tensor)
    d = tensor.detach().cpu().view(-1)
    if (floatType == 'float16'):
        d = d.to(torch.float16).numpy().astype(np.float16)
        b = struct.pack(f'{len(d)}e', *d)
        file.write(b)
    elif (floatType == 'float32'):
        d = d.to(torch.float32).numpy().astype(np.float32)
        b = struct.pack(f'{len(d)}f', *d)
        file.write(b)
    elif (floatType == 'q40'):
        d = d.to(torch.float32).numpy().astype(np.float32)
        quantizeQ40(d, file)
    else:
        raise Exception('Unknown float type')

def toDict(models):
    stateDict = {}
    for name in list(models[0]):
        tensors = [model[name] for model in models]
        if len(tensors) == 1 or len(tensors[0].shape) == 1:
            stateDict[name] = tensors[0]
            continue
        is_axis_1 = (
            name.startswith('tok_embeddings.')
            or name.endswith('.attention.wo.weight')
            or name.endswith('.feed_forward.w2.weight')
        )
        axis = 1 if is_axis_1 else 0
        stateDict[name] = torch.cat(tensors, dim=axis)
        for model in models:
            del model[name]
    return stateDict

def writeHeader(outFile, p):
    header = struct.pack('iiiiiii',
        p['dim'],
        p['hidden_dim'],
        p['n_layers'],
        p['n_heads'],
        p['n_kv_heads'],
        p['vocab_size'],
        p['max_seq_len'])
    outFile.write(header)
    return len(header)

def getBytes(nNumbers, targetFloatType):
    if (targetFloatType == 'float32'):
        return int(nNumbers * 4)
    if (targetFloatType == 'float16'):
        return int(nNumbers * 2)
    if (targetFloatType == 'q40'):
        return int((nNumbers // 32) * 18)
    raise Exception('Unknown float type')

def getBlockOffsets(params, targetFloatType):
    kvDim = (params['dim'] * params['n_kv_heads']) / params['n_heads']
    rms = getBytes(params['dim'], 'float32')
    ffn = getBytes(params['dim'], 'float32')
    q = getBytes(params['dim'] * params['dim'], targetFloatType)
    k = getBytes(params['dim'] * kvDim, targetFloatType)
    v = getBytes(params['dim'] * kvDim, targetFloatType)
    wo = getBytes(params['dim'] * params['dim'], targetFloatType)
    w1 = getBytes(params['dim'] * params['hidden_dim'], targetFloatType)
    w2 = getBytes(params['hidden_dim'] * params['dim'], targetFloatType)
    w3 = getBytes(params['dim'] * params['hidden_dim'], targetFloatType)
    return {
        'attention_norm.weight': 0,
        'ffn_norm.weight': rms,
        'attention.wq.weight': rms + ffn,
        'attention.wk.weight': rms + ffn + q,
        'attention.wv.weight': rms + ffn + k + q,
        'attention.wo.weight': rms + ffn + v + k + q,
        'feed_forward.w1.weight': rms + ffn + wo + v + k + q,
        'feed_forward.w2.weight': rms + ffn + w1 + wo + v + k + q,
        'feed_forward.w3.weight': rms + ffn + w2 + w1 + wo + v + k + q,
        '_total': rms + ffn + w3 + w2 + w1 + wo + v + k + q
    }

def convert(modelPath, outputPath, targetFloatType):
    paramsPath = os.path.join(modelPath, 'params.json')
    with open(paramsPath) as f:
        params = json.load(f)
        if (params['vocab_size'] < 1):
            raise Exception('Invalid vocab size')
        params['n_kv_heads'] = params.get('n_kv_heads') or params['n_heads']
        params['head_size'] = params['dim'] / params['n_heads']
        params['max_seq_len'] = 2048
        print(params)

    outFile = open(outputPath, 'wb')

    tokenEmbeddingBytes = getBytes(params['vocab_size'] * params['dim'], 'float32')
    rmsFinalBytes = getBytes(params['dim'], 'float32')

    isHeaderWritten = False
    modelPaths = sorted(list(Path(modelPath).glob('consolidated.*.pth')))
    for modelPath in modelPaths:
        model = torch.load(modelPath, map_location='cpu')
        modelDict = toDict([model])
        if (not isHeaderWritten):
            params['hidden_dim'] = modelDict['layers.0.feed_forward.w1.weight'].shape[0]
            headerOffset = writeHeader(outFile, params)
            blockOffsets = getBlockOffsets(params, targetFloatType)
            afterBlocksOffset = int(headerOffset + tokenEmbeddingBytes + blockOffsets['_total'] * params['n_layers'])
            ropeBytes = int(getBytes(params['max_seq_len'] * params['head_size'] / 2, 'float32') * 2)
            isHeaderWritten = True

        for layerName in modelDict.keys():
            tensor = modelDict[layerName]
            nameParts = layerName.split('.', 2)
            print(f'Exporting {layerName}...')
            if (nameParts[0] == 'layers'):
                index = int(nameParts[1])
                layerOffset = blockOffsets[nameParts[2]]
                tensorOffset = int(headerOffset + tokenEmbeddingBytes + blockOffsets['_total'] * index + layerOffset)
                tensorFloatType = 'float32' if (nameParts[2] == 'attention_norm.weight' or nameParts[2] == 'ffn_norm.weight') else targetFloatType
                outFile.seek(tensorOffset)
                exportTensor(outFile, tensor, tensorFloatType)
            elif (layerName == 'tok_embeddings.weight'):
                outFile.seek(headerOffset)
                exportTensor(outFile, tensor, 'float32')
            elif (layerName == 'norm.weight'):
                outFile.seek(afterBlocksOffset)
                exportTensor(outFile, tensor, 'float32')
            elif (layerName == 'rope.freqs'):
                # We skip this layer
                pass
            elif (layerName == 'output.weight'):
                tensorOffset = int(afterBlocksOffset + rmsFinalBytes + ropeBytes)
                outFile.seek(tensorOffset)
                exportTensor(outFile, tensor, targetFloatType)
            else:
                raise Exception(f'Unknown layer: {layerName}')

    outFile.close()

def usage():
    print('Usage: python convert.py <modelPath> <targetFloatType>')
    exit(1)

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        usage()

    modelPath = sys.argv[1]
    targetFloatType = sys.argv[2]

    if (not modelPath or not targetFloatType in ['float16', 'float32', 'q40']):
        usage()

    modelName = modelPath.split('/')[-1]
    outputFileName = f'dllama_{modelName}_{targetFloatType}.bin'

    print(f'Model name: {modelName}')
    print(f'Target float type: {targetFloatType}')
    print(f'Target file: {outputFileName}')

    convert(modelPath, outputFileName, targetFloatType)
