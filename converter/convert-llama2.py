import os
import sys
import struct
import json
import torch
import math
import time
import numpy as np
from pathlib import Path

LAYER_CHUNK_SIZE = 48

def writeQuantizedQ40Tensor(file, x):
    t0 = time.time()
    x = x.to(torch.float32).numpy().astype(np.float32)
    blockSize = 32
    blockHalfSize = blockSize // 2
    assert(x.shape[0] % blockSize == 0)
    groups = x.reshape(-1, blockSize)
    gmax = np.max(groups, axis=1)
    gmin = np.min(groups, axis=1)
    deltas = np.divide(np.where(-gmin > gmax, gmin, gmax), -8)
    deltas16 = deltas.astype(np.float16)
    ids = np.where(deltas != 0, 1.0 / deltas, 0)
    groups = np.add(groups * ids[:, np.newaxis], 8.5)
    groups = np.where(groups < 15, groups, 15)

    nBytes = 0
    block = [0] * blockHalfSize
    for groupIndex in range(0, len(groups)):
        group = groups[groupIndex]
        delta16 = deltas16[groupIndex]

        for i in range(0, blockHalfSize):
            x0 = int(group[i])
            x1 = int(group[i + blockHalfSize])
            block[i] = (x0 & 0xF) | ((x1 & 0xF) << 4)

        buffer = struct.pack(f'e{blockHalfSize}B', delta16, *block)
        file.write(buffer)
        nBytes += len(buffer)
    t1 = time.time()
    print(f'Quantized tensor to {nBytes} bytes in {t1 - t0:.2f} s')

def writeTensor(file, tensor, floatType):
    d = tensor.detach().cpu().view(-1)
    if (floatType == 'f16'):
        d = d.to(torch.float16).numpy().astype(np.float16)
        b = struct.pack(f'{len(d)}e', *d)
        file.write(b)
    elif (floatType == 'f32'):
        d = d.to(torch.float32).numpy().astype(np.float32)
        b = struct.pack(f'{len(d)}f', *d)
        file.write(b)
    elif (floatType == 'q40'):
        writeQuantizedQ40Tensor(file, d)
    else:
        raise Exception('Unknown float type')

def writeHeader(file, params):
    header = struct.pack('iiiiiii',
        params['dim'],
        params['hidden_dim'],
        params['n_layers'],
        params['n_heads'],
        params['n_kv_heads'],
        params['vocab_size'],
        params['max_seq_len'])
    file.write(header)
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
    nSlices = len(modelPaths)

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

    nChunks = math.ceil(len(layers) / LAYER_CHUNK_SIZE)
    for chunkIndex in range(0, nChunks):
        chunkLayerNames = layers[LAYER_CHUNK_SIZE * chunkIndex:LAYER_CHUNK_SIZE * (chunkIndex + 1)]
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
                params['hidden_dim'] = model['layers.0.feed_forward.w1.weight'].shape[0] * nSlices
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
            floatType = 'f32' if isAlwaysF32 else targetFloatType

            tensors = models[layerName]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                tensor = tensors[0]
            else:
                tensor = torch.cat(tensors, dim=(1 if isAxis1 else 0))

            print(f'🔶 Exporting {layerName} {tensor.shape}...')
            writeTensor(outFile, tensor, floatType)

        del models

    outFile.close()

def usage():
    print('Usage: python convert-llama2.py <modelPath> <targetFloatType>')
    exit(1)

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        usage()

    modelPath = sys.argv[1]
    targetFloatType = sys.argv[2]

    if (not modelPath or not targetFloatType in ['f16', 'f32', 'q40']):
        usage()

    modelName = modelPath.split('/')[-1]
    outputFileName = f'dllama_{modelName}_{targetFloatType}.bin'

    print(f'Model name: {modelName}')
    print(f'Target float type: {targetFloatType}')
    print(f'Target file: {outputFileName}')

    convert(modelPath, outputFileName, targetFloatType)

    print('Done!')
