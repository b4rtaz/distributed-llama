import os
import sys
import json
import torch
import math
import numpy as np
from writer import writeTensor, writeHeader, isFloatTypeSupported
from pathlib import Path
import safetensors

LAYER_CHUNK_SIZE = 48

def convert_safetensor(modelPath, outputPath, targetFloatType):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(modelPath)
    if "LlamaForCausalLM" not in config.architectures:
        print("Cannot convert as this model is not of the Llama architecture")
        return
    params = {}
    params['head_size'] = config.hidden_size / config.num_attention_heads
    params["n_layers"] = config.num_hidden_layers
    params["n_heads"] = config.num_attention_heads
    params["n_kv_heads"] = config.num_key_value_heads
    params['max_seq_len'] = config.max_position_embeddings
    params['rope_theta'] = config.rope_theta
    params['arch_type'] = 0xABCD00
    params['n_experts'] = 0
    params['n_active_experts'] = 0
    params['rope_theta'] = int(config.rope_theta)
    layers = []
    model_files = []
    for model_file in list(Path(modelPath).glob('*.safetensors')):
        if model_file.name.endswith("tokenizer.safetensors"):
            continue
        model_files.append(model_file)
        with safetensors.safe_open(model_file, framework="pt") as f:
            for layer in f.keys():
                layers.append({
                    "name" : layer,
                    "file" : model_file
                })
    print(f"Total layers: {len(layers)}")
    nChunks = math.ceil(len(layers) / LAYER_CHUNK_SIZE)
    print(f"Total chunks: {nChunks}")
        
    outFile = open(outputPath, 'wb')
    
    writeHeader(outFile, params)
        
    for chunkIndex in range(0, nChunks):
        chunkLayers = layers[LAYER_CHUNK_SIZE * chunkIndex:LAYER_CHUNK_SIZE * (chunkIndex + 1)]
        print(f'💿 Chunking model {chunkIndex + 1}/{nChunks}...')
            
        for layer in chunkLayers:
            
            if layer["name"] == 'rope.freqs':
                continue

            isAxis1 = (
                layer["name"] == 'tok_embeddings.weight' or
                layer["name"].endswith('.attention.wo.weight') or
                layer["name"].endswith('.feed_forward.w2.weight')
            )
            isAlwaysF32 = (
                layer["name"] == 'tok_embeddings.weight' or
                layer["name"].endswith('.attention_norm.weight') or
                layer["name"].endswith('.ffn_norm.weight') or
                layer["name"] == 'norm.weight'
            )
            floatType = 'f32' if isAlwaysF32 else targetFloatType

            print(f"Loading tensors for "+layer["name"]+" from: " + os.path.basename(layer["file"]))
            
            with safetensors.safe_open(layer["file"], framework="pt") as f:
                tensors = f.get_tensor(layer["name"])
                if len(tensors.shape) == 1 or len(tensors[0].shape) == 1:
                    tensor = tensors
                else:
                    tensor = torch.cat(tensors, dim=(1 if isAxis1 else 0))

                print(f'🔶 Exporting {layer["name"]} {tensor.shape}...')
                writeTensor(outFile, tensor, floatType)
            
        del models
        
    outFile.close()

def convert_pth(modelPath, outputPath, targetFloatType):
    paramsPath = os.path.join(modelPath, 'params.json')
    with open(paramsPath) as f:
        params = json.load(f)
        if (params['vocab_size'] < 1):
            raise Exception('vocab_size is invalid, please update params.json file')
        if (params.get('max_seq_len') is None):
            raise Exception('max_seq_len is required, please update params.json file')
        params['n_kv_heads'] = params.get('n_kv_heads') or params['n_heads']
        params['head_size'] = params['dim'] / params['n_heads']
        params['arch_type'] = 0xABCD00
        params['n_experts'] = 0
        params['n_active_experts'] = 0
        if ('rope_theta' in params):
            params['rope_theta'] = int(params['rope_theta'])

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

def convert(modelPath, outputPath, targetFloatType):
    model_files = list(Path(modelPath).glob('*.safetensors'))
    if len(model_files) > 0:
        convert_safetensor(modelPath, outputPath, targetFloatType)
        return
    
    model_files = list(Path(modelPath).glob('*.pth'))
    if len(model_files) > 0:
        convert_pth(model_files, outputPath, targetFloatType)
        return
    
    print("No llama could be found in: {modelPath}")

def usage():
    print('Usage: python convert-llama.py <modelFolder> <outputFolder> <targetFloatType>')
    exit(1)

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        usage()

    modelPath = sys.argv[1]
    outputPath = sys.argv[2]
    targetFloatType = sys.argv[3].lower()

    if (not modelPath or not isFloatTypeSupported(targetFloatType)):
        usage()

    modelName = os.path.basename(modelName)
        
    outputFileName = f'dllama_{modelName.lower()}_{targetFloatType}.bin'

    print(f'Model name: {modelName}')
    print(f'Target float type: {targetFloatType}')
    print(f'Target file: {outputFileName}')

    convert(modelPath, outputFileName, targetFloatType)

    print('Done!')
