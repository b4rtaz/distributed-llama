import gc
import torch
import sys
import os
from writer import isFloatTypeSupported, writeTensor, writeHeader 

# Model: https://huggingface.co/keyfan/grok-1-hf/tree/main

currentFileIndex = None
model = None
layerMap = {}
folderPath = None

def unloadModel():
    global model
    if model:
       del model
       model = None
       gc.collect()

def loadModel(index):
    global currentFileIndex
    global model
    global layerMap
    global folderPath
    if (currentFileIndex == index):
        return
    unloadModel()
    fileName = f'pytorch_model-000{str(index).zfill(2)}-of-00019.bin'
    filePath = os.path.join(folderPath, fileName)
    print(f'💿 Loading file {fileName}...')
    model = torch.load(filePath, map_location='cpu')
    layerNames = list(model.keys())
    for layerName in layerNames:
        layerMap[layerName] = index
    print(f'Found layers: {layerNames}')
    currentFileIndex = index

def writeLayer(outFile, layerName, targetFloatType):
    global currentFileIndex
    global model
    global layerMap

    if (not layerName in model):
        if (layerName in layerMap):
            loadModel(layerMap[layerName])
        else:
            loadModel(currentFileIndex + 1)
    if (not layerName in model):
        raise Exception(f'Cannot load {layerName}')

    tensor = model[layerName]
    print(f'🔶 Writing tensor {layerName} {tensor.shape}...')
    writeTensor(outFile, tensor, targetFloatType)

def convert(targetFloatType, outputFileName):
    outFile = open(outputFileName, 'wb')

    params = {
        'arch_type': 0xABCD01,
        'dim': 6144,
        'hidden_dim': 32768,
        'n_layers': 64,
        'n_heads': 48,
        'n_kv_heads': 8,
        'n_experts': 8,
        'n_active_experts': 2,
        'vocab_size': 131072,
        'max_seq_len': 8192,
    }
    writeHeader(outFile, params)

    #### pytorch_model-00001-of-00019.bin -> pytorch_model-00019-of-00019.bin
    loadModel(1)

    writeLayer(outFile, 'transformer.in_out_embed.weight', 'f32')

    for index in range(0, params['n_layers']):
        writeLayer(outFile, f'transformer.decoder_layer.{index}.multi_head_attention.query.weight', targetFloatType)
        writeLayer(outFile, f'transformer.decoder_layer.{index}.multi_head_attention.key.weight', targetFloatType)
        writeLayer(outFile, f'transformer.decoder_layer.{index}.multi_head_attention.value.weight', targetFloatType)
        writeLayer(outFile, f'transformer.decoder_layer.{index}.multi_head_attention.linear.weight', targetFloatType)

        writeLayer(outFile, f'transformer.decoder_layer.{index}.router.weight', targetFloatType)
        for e in range(params['n_experts']):
            writeLayer(outFile, f'transformer.decoder_layer.{index}.moe.{e}.linear_v.weight', targetFloatType) # up
            writeLayer(outFile, f'transformer.decoder_layer.{index}.moe.{e}.linear.weight', targetFloatType) # gate
            writeLayer(outFile, f'transformer.decoder_layer.{index}.moe.{e}.linear_1.weight', targetFloatType) # down

        writeLayer(outFile, f'transformer.decoder_layer.{index}.rms_norm.weight', 'f32')
        writeLayer(outFile, f'transformer.decoder_layer.{index}.rms_norm_1.weight', 'f32')
        writeLayer(outFile, f'transformer.decoder_layer.{index}.rms_norm_2.weight', 'f32')
        writeLayer(outFile, f'transformer.decoder_layer.{index}.rms_norm_3.weight', 'f32')

    #### pytorch_model-00019-of-00019.bin
    loadModel(19)

    writeLayer(outFile, 'transformer.rms_norm.weight', 'f32') # rmsFinalNorm

    #### pytorch_model-00001-of-00019.bin
    loadModel(1)

    writeLayer(outFile, 'lm_head.weight', targetFloatType)

    unloadModel()

    outFile.close()
    print(f'Converted {outputFileName}')

def usage():
    print('Usage: python convert-grok-1.py <modelPath> <targetFloatType>')
    exit(1)

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        usage()

    folderPath = sys.argv[1]
    targetFloatType = sys.argv[2]
    outputFileName = f'dllama-grok-1-{targetFloatType}.bin'

    if not isFloatTypeSupported(targetFloatType):
        print('Float type is not supported')
        exit(1)

    convert(targetFloatType, outputFileName)
