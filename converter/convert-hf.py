import gc
import json
import sys
import os
from writer import parseFloatType, writeTensor, writeHeader, FloatType
from safetensors import safe_open

def permute(tensor, nHeads: int, nKvHeads: int):
    if nHeads != nKvHeads:
        nHeads = nKvHeads
    return (tensor.reshape(nHeads, 2, tensor.shape[0] // nHeads // 2, *tensor.shape[1:]).swapaxes(1, 2).reshape(tensor.shape))

class Processor:
    def __init__(self, config):
        self.config = config
        self.currentModelIndex = None
        self.currentModel = None
        self.currentModelKeys = None
        self.layerMap = {}
        self.plan = []

    def __unloadModel(self):
        if self.currentModel:
            del self.currentModel
            self.currentModel = None
            gc.collect()

    def __loadModel(self, index):
        if (self.currentModelIndex == index):
            return
        self.__unloadModel()
        filePath = self.config['files'][index]
        fileName = os.path.basename(filePath)
        print(f'💿 Loading file {fileName}...')
        self.currentModel = safe_open(filePath, framework='pt', device='cpu')
        self.currentModelKeys = list(self.currentModel.keys())
        for key in self.currentModelKeys:
            self.layerMap[key] = index
        print(f'Found {len(self.currentModelKeys)} layers')
        self.currentModelIndex = index

    def __permuteQ(self, tensor):
        return permute(tensor, self.config['n_heads'], self.config['n_heads'])

    def __permuteK(self, tensor):
        return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])

    def __preparePlan(self):
        floatType = self.config['weights_float_type']
        p = self.plan
        p.append([FloatType.F32, 'model.embed_tokens.weight'])
        for l in range(0, self.config['n_layers']):
            p.append([floatType, self.__permuteQ, f'model.layers.{l}.self_attn.q_proj.weight'])
            p.append([floatType, self.__permuteK, f'model.layers.{l}.self_attn.k_proj.weight'])
            p.append([floatType, f'model.layers.{l}.self_attn.v_proj.weight'])
            p.append([floatType, f'model.layers.{l}.self_attn.o_proj.weight'])

            if (self.config['n_experts'] > 0):
                for e in range(self.config['n_experts']):
                    p.append([floatType, f'model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight']) # up
                    p.append([floatType, f'model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight']) # gate
                    p.append([floatType, f'model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight']) # down
            else:
                p.append([floatType, f'model.layers.{l}.mlp.gate_proj.weight']) # gate
                p.append([floatType, f'model.layers.{l}.mlp.down_proj.weight']) # down
                p.append([floatType, f'model.layers.{l}.mlp.up_proj.weight']) # up

            p.append([FloatType.F32, f'model.layers.{l}.input_layernorm.weight'])
            p.append([FloatType.F32, f'model.layers.{l}.post_attention_layernorm.weight'])
        p.append([FloatType.F32, 'model.norm.weight'])
        p.append([floatType, 'lm_head.weight'])

    def write(self, outputFile):
        self.__preparePlan()
        for planItem in self.plan:
            lookup = planItem[1:]
            transform = None
            if (callable(lookup[0])):
                transform = lookup[0]
                lookup = lookup[1:]

            if (self.currentModelIndex == None):
                modelIndex = 0
            else:
                modelIndex = None
                for layerName in lookup:
                    if (layerName in self.layerMap):
                        modelIndex = self.layerMap[layerName]
                        break
                if (modelIndex is None):
                    modelIndex = self.currentModelIndex + 1
            self.__loadModel(modelIndex)

            tensor = None
            for layerName in lookup:
                if (layerName in self.currentModelKeys):
                    tensor = self.currentModel.get_tensor(layerName)
                    break
            if tensor is None:
                raise Exception(f'Layer {lookup[0]} not found')
            print(f'🔶 Writing tensor {layerName} {tensor.shape}...')

            floatType = planItem[0]
            if (transform):
                tensor = transform(tensor)
            writeTensor(outputFile, tensor, floatType)

def loadConfig(folderPath, weightsFloatType):
    allFiles = os.listdir(folderPath)
    allFiles.sort()
    with open(os.path.join(folderPath, 'config.json')) as fc:
        config = json.load(fc)
    files = []
    for fileName in allFiles:
        if fileName.endswith('.safetensors'):
            files.append(os.path.join(folderPath, fileName))
    nFiles = len(files)
    if (nFiles == 0):
        raise Exception('Not found any model file')

    archType = {
        'llama': 0xABCD00,
    }.get(config['model_type'])
    if (archType is None):
        raise Exception('Unknown arch_type')
    hiddenAct = {
        'gelu': 0,
        'silu': 1
    }.get(config['hidden_act'])
    if (hiddenAct is None):
        raise Exception('Unknown hidden_act')

    result = {
        'version': 0,
        'arch_type': archType,
        'dim': config['hidden_size'],
        'hidden_dim': config['intermediate_size'],
        'n_layers': config['num_hidden_layers'],
        'n_heads': config['num_attention_heads'],
        'n_kv_heads': config['num_key_value_heads'],
        'weights_float_type': weightsFloatType,
        'max_seq_len': config['max_position_embeddings'],
        'vocab_size': config['vocab_size'],
        'hidden_act': hiddenAct,
        'files': files,
        'n_files': nFiles,
    }
    if ('rope_theta' in config):
        config['rope_theta'] = int(config['rope_theta'])
    if ('num_local_experts' in config):
        result['n_experts'] = config['num_local_experts']
        result['n_active_experts'] = config['num_active_local_experts']
    else:
        result['n_experts'] = 0
        result['n_active_experts'] = 0
    return result

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        print('Usage: python convert-hf.py <sourceFolderPath> <weightsFloatType> <outputName>')
        exit(1)

    sourceFolderPath = sys.argv[1]
    weightsFloatType = parseFloatType(sys.argv[2])
    outputFileName = f'dllama_{sys.argv[3]}_{sys.argv[2]}.bin'

    print(f'Output file: {outputFileName}')

    config = loadConfig(sourceFolderPath, weightsFloatType)

    with open(outputFileName, 'wb') as outputFile:
        writeHeader(outputFile, config)
        processor = Processor(config)
        processor.write(outputFile)

    print(f'✅ {outputFileName} created successfully')