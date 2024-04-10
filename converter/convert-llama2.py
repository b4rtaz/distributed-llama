import os
import sys
import json
import torch
import math
import numpy as np
from writer import writeTensor, writeHeader 
from pathlib import Path

LAYER_CHUNK_SIZE = 48

def convert(modelPath, outputPath, targetFloatType):
    paramsPath = os.path.join(modelPath, 'params.json')
    with open(paramsPath) as f:
        params = json.load(f)
        if (params['vocab_size'] < 1):
            raise Exception('Invalid vocab size')
        params['n_kv_heads'] = params.get('n_kv_heads') or params['n_heads']
        params['head_size'] = params['dim'] / params['n_heads']
        params['max_seq_len'] = 2048
        params['arch_type'] = 0xABCD00
        params['n_experts'] = 0
        params['n_active_experts'] = 0

    modelPaths = sorted(list(Path(modelPath).glob('consolidated.*.pth')))
    nSlices = len(modelPaths)

    layers = []
    layers.append('tok_embeddings.weight')
    for layerIndex in range(0, params['n_layers']):
        layers.append(f'layers.{layerIndex}.attention.wq.weight')
        layers.append(f'layers.{layerIndex}.attention.wk.weight')
        layers.append(f'layers.{layerIndex}.attention.wv.weight')
        layers.append(f'layers.{layerIndex}.attention.wo.weight')
        layers.append(f'layers.{layerIndex}.feed_forward.w1.weight')
        layers.append(f'layers.{layerIndex}.feed_forward.w2.weight')
        layers.append(f'layers.{layerIndex}.feed_forward.w3.weight')
        layers.append(f'layers.{layerIndex}.attention_norm.weight')
        layers.append(f'layers.{layerIndex}.ffn_norm.weight')
    layers.append('norm.weight')
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
