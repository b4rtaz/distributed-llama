import gc
import sys
import os
from writer import isFloatTypeSupported, writeTensor, writeHeader
from safetensors import safe_open

# Model: https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1

currentFileIndex = None
nFiles = None
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
    global nFiles
    global model
    global layerMap
    global folderPath
    if (currentFileIndex == index):
        return
    unloadModel()
    fileName = f'model-000{str(index).zfill(2)}-of-000{nFiles}.safetensors'# + '?download=true'
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

def getParams(modelName):
    params = {
        'arch_type': 0xABCD02,
        'vocab_size': 32000,
        'n_experts': 8,
        'n_active_experts': 2,
        'hidden_act': 1, # silu
        'rope_theta': 1000000,
    }
    if (modelName == '8x7B'):
        params['dim'] = 4096
        params['hidden_dim'] = 14336
        params['n_layers'] = 32
        params['n_heads'] = 32
        params['n_kv_heads'] = 8
        params['max_seq_len'] = 32768
        params['n_files'] = 19
    elif (modelName == '8x22B'):
        params['dim'] = 6144
        params['hidden_dim'] = 16384
        params['n_layers'] = 56
        params['n_heads'] = 48
        params['n_kv_heads'] = 8
        params['max_seq_len'] = 65536
        params['n_files'] = 59
    else:
        raise Exception(f'Unknown model {modelName}')
    return params

def convert(modelName, targetFloatType, outputFileName):
    global nFiles
    params = getParams(modelName)
    nFiles = params['n_files']

    outFile = open(outputFileName, 'wb')
    writeHeader(outFile, params)

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

    loadModel(nFiles)

    writeLayer(outFile, 'model.norm.weight', 'f32')
    writeLayer(outFile, 'lm_head.weight', targetFloatType)

    unloadModel()

    outFile.close()
    print(f'Converted {outputFileName}')

def usage():
    print('Usage: python convert-mixtral.py <modelName> <modelPath> <targetFloatType>')
    exit(1)

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        usage()

    modelName = sys.argv[1]
    folderPath = sys.argv[2]
    targetFloatType = sys.argv[3]
    outputFileName = f'dllama_mixtral_{modelName.lower()}-{targetFloatType}.bin'

    if not isFloatTypeSupported(targetFloatType):
        print('Float type is not supported')
        exit(1)

    convert(modelName, targetFloatType, outputFileName)
