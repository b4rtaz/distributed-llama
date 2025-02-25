import gc
import json
import sys
import os
from writer import parseFloatType, writeTensor, writeHeader, FloatType
from safetensors import safe_open

class ArchType:
    LLAMA = 0xABCD00

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

    def __loadModel(self, index: int):
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
        wt = self.config['weights_float_type']
        p = self.plan
        p.append([FloatType.F32,
            'model.embed_tokens.weight'])
        for l in range(0, self.config['n_layers']):
            p.append([wt, self.__permuteQ,
                f'model.layers.{l}.self_attn.q_proj.weight'])
            p.append([wt, self.__permuteK,
                f'model.layers.{l}.self_attn.k_proj.weight'])
            p.append([wt,
                f'model.layers.{l}.self_attn.v_proj.weight'])
            p.append([wt,
                f'model.layers.{l}.self_attn.o_proj.weight'])

            if (self.config['n_experts'] > 0):
                for e in range(self.config['n_experts']):
                    p.append([wt,
                        f'model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight']) # up
                    p.append([wt,
                        f'model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight']) # gate
                    p.append([wt,
                        f'model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight']) # down
            else:
                p.append([wt,
                    f'model.layers.{l}.mlp.gate_proj.weight']) # gate
                p.append([wt,
                    f'model.layers.{l}.mlp.down_proj.weight']) # down
                p.append([wt,
                    f'model.layers.{l}.mlp.up_proj.weight']) # up

            p.append([FloatType.F32,
                f'model.layers.{l}.input_layernorm.weight'])
            p.append([FloatType.F32,
                f'model.layers.{l}.post_attention_layernorm.weight'])
        p.append([FloatType.F32,
            'model.norm.weight'])
        p.append([wt,
            'lm_head.weight', 'model.embed_tokens.weight'])

    def write(self, outputFile: str):
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

def parseArchType(type: str):
    archType = {
        'llama': ArchType.LLAMA,
        'mistral': ArchType.LLAMA,
    }.get(type)
    if (archType is None):
        raise Exception(f'Unsupported arch type: {type}')
    return archType

def parseHiddenAct(act: str):
    hiddenAct = {
        'gelu': 0,
        'silu': 1
    }.get(act)
    if (hiddenAct is None):
        raise Exception(f'Unsupported hidden act: {act}')
    return hiddenAct

def parseRopeType(rt: str):
    ropeType = {
        'llama3': 2, # LLAMA3_1
    }.get(rt)
    if (ropeType is None):
        raise Exception(f'Unsupported rope type: {ropeType}')
    return ropeType

def loadConfig(folderPath: str, weightsFloatType: int):
    allFiles = os.listdir(folderPath)
    allFiles.sort()
    with open(os.path.join(folderPath, 'config.json')) as fc:
        config = json.load(fc)
    files = []
    for fileName in allFiles:
        if fileName.endswith('.safetensors') and not fileName.startswith('.'):
            files.append(os.path.join(folderPath, fileName))
    if (len(files) == 0):
        raise Exception('Not found any model file')

    result = {
        'version': 0,
        'arch_type': parseArchType(config['model_type']),
        'hidden_act': parseHiddenAct(config['hidden_act']),
        'dim': config['hidden_size'],
        'hidden_dim': config['intermediate_size'],
        'n_layers': config['num_hidden_layers'],
        'n_heads': config['num_attention_heads'],
        'n_kv_heads': config['num_key_value_heads'],
        'weights_float_type': weightsFloatType,
        'max_seq_len': config['max_position_embeddings'],
        'vocab_size': config['vocab_size'],
        'files': files,
    }

    nExperts = config.get('num_local_experts')
    nActiveExperts = config.get('num_active_local_experts') or config.get('num_experts_per_tok')
    result['n_experts'] = int(nExperts) if nExperts is not None else 0
    result['n_active_experts'] = int(nActiveExperts) if nActiveExperts is not None else 0

    ropeTheta = config.get('rope_theta')
    if (ropeTheta is not None):
        result['rope_theta'] = int(ropeTheta)

    ropeScaling = config.get('rope_scaling')
    if (ropeScaling is not None):
        result['rope_scaling_factor'] = int(ropeScaling['factor'])
        result['rope_scaling_low_freq_factor'] = int(ropeScaling['low_freq_factor'])
        result['rope_scaling_high_freq_factory'] = int(ropeScaling['high_freq_factor'])
        result['rope_scaling_orig_max_seq_len'] = int(ropeScaling['original_max_position_embeddings'])
        result['rope_type'] = parseRopeType(ropeScaling['rope_type'])
    return result

def printUsage():
    print('Usage: python convert-hf.py <sourceFolderPath> <weightsFloatType> <name>')
    print()
    print('Options:')
    print('  <sourceFolderPath> The path to the folder containing the model files')
    print('  <weightsFloatType> The float type of the weights (e.g. "q40")')
    print('  <name>             The name of the model (e.g. "llama3")')

if __name__ == '__main__':
    if (len(sys.argv) < 4):
        printUsage()
        exit(1)

    sourceFolderPath = sys.argv[1]
    weightsFloatType = parseFloatType(sys.argv[2])
    name = sys.argv[3]
    outputFileName = f'dllama_model_{name}_{sys.argv[2]}.m'

    print(f'Output file: {outputFileName}')

    config = loadConfig(sourceFolderPath, weightsFloatType)

    with open(outputFileName, 'wb') as outputFile:
        writeHeader(outputFile, config)
        processor = Processor(config)
        processor.write(outputFile)

    print(f'✅ {outputFileName} created successfully')