import os
import sys
import struct
import json
import torch
import math
import numpy as np
from pathlib import Path

def writeQuantizedQ40Tensor(x, file):
    groupSize = 32
    groupSizeHalf = groupSize // 2
    assert(x.shape[0] % groupSize == 0)
    groups = x.reshape(-1, groupSize)
    gmax = np.max(groups, axis=1)
    gmin = np.min(groups, axis=1)
    absMax = np.where(-gmin > gmax, gmin, gmax)

    nBytes = 0
    for gi, group in enumerate(groups):
        groupAbsMax = absMax[gi]
        delta = groupAbsMax / -8
        id = (1.0 / delta) if delta else 0

        twoBytes = struct.pack('e', np.float16(delta).astype(np.float16))
        file.write(twoBytes)
        nBytes += 2

        for i in range(0, groupSizeHalf):
            x0 = group[i] * id + 8.5
            x1 = group[i + groupSizeHalf] * id + 8.5
            xi0 = min(15, int(x0))
            xi1 = min(15, int(x1))
            b = (xi0 & 0xF) | ((xi1 & 0xF) << 4)
            byte = struct.pack('B', b)
            file.write(byte)
            nBytes += 1
    print(f'Quantized {x.shape[0] * 4} bytes into {nBytes} bytes')

def writeTensor(file, tensor, floatType):
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
        writeQuantizedQ40Tensor(d, file)
    else:
        raise Exception('Unknown float type')

def writeHeader(outFile, params):
    header = struct.pack('iiiiiii',
        params['dim'],
        params['hidden_dim'],
        params['n_layers'],
        params['n_heads'],
        params['n_kv_heads'],
        params['vocab_size'],
        params['max_seq_len'])
    outFile.write(header)
    print(params)

def convert(modelPath, outputPath, targetFloatType):
    paramsPath = os.path.join(modelPath, 'params.json')
    with open(paramsPath) as f:
        params = json.load(f)
        if (params['vocab_size'] < 1):
            raise Exception('Invalid vocab size')
        params['n_kv_heads'] = params.get('n_kv_heads') or params['n_heads']
        params['head_size'] = params['dim'] / params['n_heads']
        params['max_seq_len'] = 2048

    modelPaths = sorted(list(Path(modelPath).glob('consolidated.*.pth')))
    nModels = len(modelPath)
    layers = []
    layers.append('tok_embeddings.weight')
    for layerIndex in range(0, params['n_layers']):
        layers.append(f'layers.{layerIndex}.attention_norm.weight')
        layers.append(f'layers.{layerIndex}.ffn_norm.weight')
        layers.append(f'layers.{layerIndex}.attention.wq.weight')
        layers.append(f'layers.{layerIndex}.attention.wk.weight')
        layers.append(f'layers.{layerIndex}.attention.wv.weight')
        layers.append(f'layers.{layerIndex}.attention.wo.weight')
        layers.append(f'layers.{layerIndex}.feed_forward.w1.weight')
        layers.append(f'layers.{layerIndex}.feed_forward.w2.weight')
        layers.append(f'layers.{layerIndex}.feed_forward.w3.weight')
    layers.append('norm.weight')
    layers.append('rope.freqs')
    layers.append('output.weight')

    isHeaderWrote = False
    outFile = open(outputPath, 'wb')

    chunkSize = 32
    nChunks = math.ceil(len(layers) / chunkSize)
    for chunkIndex in range(0, nChunks):
        chunkLayerNames = layers[chunkSize * chunkIndex:chunkSize * (chunkIndex + 1)]
        models = {}
        for layerName in chunkLayerNames:
            models[layerName] = []

        print(f'💿 Chunking model {chunkIndex + 1}/{nChunks}...')

        for modelPath in modelPaths:
            model = torch.load(modelPath, map_location='cpu')
            for modelKey in model:
                if (modelKey in chunkLayerNames):
                    models[modelKey].append(model[modelKey])
            if not isHeaderWrote:
                params['hidden_dim'] = model['layers.0.feed_forward.w1.weight'].shape[0] * nModels
                writeHeader(outFile, params)
                isHeaderWrote = True
            del model

        for layerName in chunkLayerNames:
            if layerName == 'rope.freqs':
                nBytes = int(params['max_seq_len'] * (params['head_size'] / 2) * 4)
                outFile.seek(nBytes, 1) # freq_cis_real (for RoPE)
                outFile.seek(nBytes, 1) # freq_cis_imag (for RoPE)
                continue

            isAxis1 = (
                layerName == 'tok_embeddings.weight' or
                layerName.endswith('.attention.wo.weight') or
                layerName.endswith('.feed_forward.w2.weight')
            )
            isAlwaysF32 = (
                layerName == 'tok_embeddings.weight' or
                layerName.endswith('.attention_norm.weight') or
                layerName.endswith('.ffn_norm.weight') or
                layerName == 'norm.weight'
            )
            floatType = 'float32' if isAlwaysF32 else targetFloatType

            tensors = models[layerName]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                tensor = tensors[0]
            else:
                tensor = torch.cat(tensors, dim=(1 if isAxis1 else 0))

            print(f'🔶 Exporting {layerName} ({tensor.shape})...')
            writeTensor(outFile, tensor, floatType)

        del models

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
    outputFileName = f'dllama_{modelName}_{targetFloatType}2.bin'

    print(f'Model name: {modelName}')
    print(f'Target float type: {targetFloatType}')
    print(f'Target file: {outputFileName}')

    convert(modelPath, outputFileName, targetFloatType)

    print('Done!')
