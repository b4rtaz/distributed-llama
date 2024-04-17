import gc
import sys
import os
from writer import isFloatTypeSupported, writeTensor, writeHeader
from safetensors import safe_open

# Model: https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1

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
    fileName = f'model-000{str(index).zfill(2)}-of-00059.safetensors' + '?download=true'
    filePath = os.path.join(folderPath, fileName)
    print(f'💿 Loading file {fileName}...')
    model = safe_open(filePath, framework='pt', device='cpu')
    layerNames = list(model.keys())
    for layerName in layerNames:
        layerMap[layerName] = index
    print(f'Found layers: {layerNames}')
    currentFileIndex = index

def writeLayer(outFile, layerName, targetFloatType):
    global currentFileIndex
    global model
    global layerMap

    if (not layerName in model.keys()):
        if (layerName in layerMap):
            loadModel(layerMap[layerName])
        else:
            loadModel(currentFileIndex + 1)
    if (not layerName in model.keys()):
        raise Exception(f'Cannot load {layerName}')

    tensor = model.get_tensor(layerName)
    print(f'🔶 Writing tensor {layerName} {tensor.shape}...')
    writeTensor(outFile, tensor, targetFloatType)

def convert(targetFloatType, outputFileName):
    outFile = open(outputFileName, 'wb')

    params = {
        'arch_type': 0xABCD02,
        'dim': 6144,
        'hidden_dim': 16384,
        'n_layers': 56,
        'n_heads': 48,
        'n_kv_heads': 8,
        'n_experts': 8,
        'n_active_experts': 2,
        'vocab_size': 32000,
        'max_seq_len': 65536,
        'hidden_act': 1, # silu
        'rope_theta': 1000000
    }
    writeHeader(outFile, params)

    #### model-00001-of-00059.safetensors -> model-00059-of-00059.safetensors
    loadModel(1)

    writeLayer(outFile, 'model.embed_tokens.weight', 'f32')

    for index in range(0, params['n_layers']):
        writeLayer(outFile, f'model.layers.{index}.self_attn.q_proj.weight', targetFloatType)
        writeLayer(outFile, f'model.layers.{index}.self_attn.k_proj.weight', targetFloatType)
        writeLayer(outFile, f'model.layers.{index}.self_attn.v_proj.weight', targetFloatType)
        writeLayer(outFile, f'model.layers.{index}.self_attn.o_proj.weight', targetFloatType)

        writeLayer(outFile, f'model.layers.{index}.block_sparse_moe.gate.weight', targetFloatType)
        for e in range(params['n_experts']):
            writeLayer(outFile, f'model.layers.{index}.block_sparse_moe.experts.{e}.w3.weight', targetFloatType) # up
            writeLayer(outFile, f'model.layers.{index}.block_sparse_moe.experts.{e}.w1.weight', targetFloatType) # gate
            writeLayer(outFile, f'model.layers.{index}.block_sparse_moe.experts.{e}.w2.weight', targetFloatType) # down

        writeLayer(outFile, f'model.layers.{index}.input_layernorm.weight', 'f32')
        writeLayer(outFile, f'model.layers.{index}.post_attention_layernorm.weight', 'f32')

    loadModel(59)

    writeLayer(outFile, 'model.norm.weight', 'f32')
    writeLayer(outFile, 'lm_head.weight', targetFloatType)

    unloadModel()

    outFile.close()
    print(f'Converted {outputFileName}')

def usage():
    print('Usage: python convert-mixtral.py <modelPath> <targetFloatType>')
    exit(1)

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        usage()

    folderPath = sys.argv[1]
    targetFloatType = sys.argv[2]
    outputFileName = f'dllama-mixtral-{targetFloatType}.bin'

    if not isFloatTypeSupported(targetFloatType):
        print('Float type is not supported')
        exit(1)

    convert(targetFloatType, outputFileName)
