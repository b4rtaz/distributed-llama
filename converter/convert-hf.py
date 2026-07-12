import datetime
import gc
import json
import os
import sys
import traceback
from writer import parseFloatType, writeTensor, writeHeader, FloatType
from safetensors import safe_open

class ArchType:
    LLAMA = 0xABCD00
    QWEN3 = 0xABCD01
    QWEN3_MOE = 0xABCD02
    QWEN3_5 = 0xABCD03

def permute(tensor, nHeads: int, nKvHeads: int):
    if nHeads != nKvHeads:
        nHeads = nKvHeads
    return (tensor.reshape(nHeads, 2, tensor.shape[0] // nHeads // 2, *tensor.shape[1:]).swapaxes(1, 2).reshape(tensor.shape))

class Processor:
    def __init__(self, config):
        self.config = config
        self.archType = config['arch_type']
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
        self.currentModelIndex = None

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

    def __transformQ(self, tensor):
        if self.archType == ArchType.LLAMA:
            return permute(tensor, self.config['n_heads'], self.config['n_heads'])
        return tensor

    def __transformK(self, tensor):
        if self.archType == ArchType.LLAMA:
            return permute(tensor, self.config['n_heads'], self.config['n_kv_heads'])
        return tensor

    def __preparePlan(self):
        if self.archType == ArchType.QWEN3_5:
            return self.__preparePlanQwen3_5()

        wt = self.config['weights_float_type']
        p = self.plan
        prefix = self.config.get('prefix', 'model')
        p.append([FloatType.F32,
            f'{prefix}.embed_tokens.weight'])
        for l in range(0, self.config['n_layers']):
            p.append([wt, self.__transformQ,
                f'{prefix}.layers.{l}.self_attn.q_proj.weight'])
            p.append([wt, self.__transformK,
                f'{prefix}.layers.{l}.self_attn.k_proj.weight'])
            p.append([wt,
                f'{prefix}.layers.{l}.self_attn.v_proj.weight'])
            p.append([wt,
                f'{prefix}.layers.{l}.self_attn.o_proj.weight'])

            if (self.config['n_experts'] > 0):
                p.append([FloatType.F32, f'{prefix}.layers.{l}.mlp.gate.weight'])
                for e in range(self.config['n_experts']):
                    p.append([wt,
                        f'{prefix}.layers.{l}.mlp.experts.{e}.gate_proj.weight'])
                    p.append([wt,
                        f'{prefix}.layers.{l}.mlp.experts.{e}.down_proj.weight'])
                    p.append([wt,
                        f'{prefix}.layers.{l}.mlp.experts.{e}.up_proj.weight'])
            else:
                p.append([wt,
                    f'{prefix}.layers.{l}.mlp.gate_proj.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.mlp.down_proj.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.mlp.up_proj.weight'])

            if (self.archType == ArchType.QWEN3 or self.archType == ArchType.QWEN3_MOE):
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.self_attn.q_norm.weight'])
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.self_attn.k_norm.weight'])

            p.append([FloatType.F32,
                f'{prefix}.layers.{l}.input_layernorm.weight'])
            p.append([FloatType.F32,
                f'{prefix}.layers.{l}.post_attention_layernorm.weight'])
        p.append([FloatType.F32,
            f'{prefix}.norm.weight'])
        p.append([wt,
            'lm_head.weight', f'{prefix}.embed_tokens.weight'])

    def __preparePlanQwen3_5(self):
        wt = self.config['weights_float_type']
        p = self.plan
        prefix = self.config.get('prefix', 'model')
        layerTypes = self.config['layer_types']

        p.append([FloatType.F32, f'{prefix}.embed_tokens.weight'])

        for l in range(self.config['n_layers']):
            lt = layerTypes[l]

            p.append([FloatType.F32,
                f'{prefix}.layers.{l}.input_layernorm.weight'])

            if lt == 'linear_attention':
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.linear_attn.A_log'])
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.linear_attn.dt_bias'])
                p.append([wt,
                    f'{prefix}.layers.{l}.linear_attn.conv1d.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.linear_attn.in_proj_qkv.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.linear_attn.in_proj_z.weight'])
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.linear_attn.in_proj_a.weight'])
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.linear_attn.in_proj_b.weight'])
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.linear_attn.norm.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.linear_attn.out_proj.weight'])
            else:
                p.append([wt,
                    f'{prefix}.layers.{l}.self_attn.q_proj.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.self_attn.k_proj.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.self_attn.v_proj.weight'])
                p.append([wt,
                    f'{prefix}.layers.{l}.self_attn.o_proj.weight'])
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.self_attn.q_norm.weight'])
                p.append([FloatType.F32,
                    f'{prefix}.layers.{l}.self_attn.k_norm.weight'])

            p.append([FloatType.F32,
                f'{prefix}.layers.{l}.post_attention_layernorm.weight'])
            p.append([wt,
                f'{prefix}.layers.{l}.mlp.gate_proj.weight'])
            p.append([wt,
                f'{prefix}.layers.{l}.mlp.down_proj.weight'])
            p.append([wt,
                f'{prefix}.layers.{l}.mlp.up_proj.weight'])

        p.append([FloatType.F32,
            f'{prefix}.norm.weight'])
        p.append([wt,
            'lm_head.weight', f'{prefix}.embed_tokens.weight'])

    def write(self, outputFile: str):
        self.__preparePlan()

        # Loading the last model file to get the layer names
        self.__loadModel(len(self.config['files']) - 1)
        self.__unloadModel()

        total = len(self.plan)
        for planIndex, planItem in enumerate(self.plan):
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
            matched_name = None
            for layerName in lookup:
                if (layerName in self.currentModelKeys):
                    tensor = self.currentModel.get_tensor(layerName)
                    matched_name = layerName
                    break
            if tensor is None:
                raise Exception(
                    f'Layer not found at plan index {planIndex}/{total}: '
                    f'lookups={list(lookup)}, '
                    f'currentModelIndex={self.currentModelIndex}, '
                    f'currentModelFile={self.config["files"][self.currentModelIndex] if self.currentModelIndex is not None else None}'
                )
            print(f'🔶 [{planIndex:4d}/{total}] Writing tensor {matched_name} {tensor.shape}...', flush=True)

            floatType = planItem[0]
            if (transform):
                tensor = transform(tensor)
            writeTensor(outputFile, tensor, floatType)

def parseArchType(type: str):
    archType = {
        'llama': ArchType.LLAMA,
        'mistral': ArchType.LLAMA,
        'qwen3': ArchType.QWEN3,
        'qwen3_moe': ArchType.QWEN3_MOE,
        'qwen3_5': ArchType.QWEN3_5,
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
        'default': 3, # Qwen multimodal RoPE
        'llama3': 2, # LLAMA3_1
    }.get(rt)
    if (ropeType is None):
        raise Exception(f'Unsupported rope type: {ropeType}')
    return ropeType

def parseRmsNormEpsilon(epsilon: float):
    if (epsilon == 1e-05):
        return 5
    elif (epsilon == 1e-06):
        return 6
    raise Exception(f'Unsupported epsilon: {epsilon}')

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

    modelType = config['model_type']

    if modelType == 'qwen3_5':
        tc = config['text_config']
        result = {
            'version': 0,
            'arch_type': parseArchType(modelType),
            'hidden_act': parseHiddenAct(tc['hidden_act']),
            'dim': tc['hidden_size'],
            'hidden_dim': tc['intermediate_size'],
            'n_layers': tc['num_hidden_layers'],
            'n_heads': tc['num_attention_heads'],
            'n_kv_heads': tc['num_key_value_heads'],
            'weights_float_type': weightsFloatType,
            'max_seq_len': tc['max_position_embeddings'],
            'vocab_size': tc['vocab_size'],
            'files': files,
            'n_experts': 0,
            'n_active_experts': 0,
            'prefix': 'model.language_model',
            'layer_types': tc['layer_types'],
        }

        ropeParams = tc.get('rope_parameters', {})
        result['rope_theta'] = int(ropeParams.get('rope_theta', 10000))
        result['rope_type'] = parseRopeType(ropeParams.get('rope_type', 'default'))

        headDim = tc.get('head_dim')
        if headDim is not None:
            result['head_dim'] = headDim

        rmsNormEps = tc.get('rms_norm_eps')
        if rmsNormEps is not None:
            result['norm_epsilon'] = parseRmsNormEpsilon(rmsNormEps)

        partialRF = tc.get('partial_rotary_factor', 1.0)
        result['partial_rotary_factor'] = int(partialRF * 100)

        attnGate = tc.get('attn_output_gate', False)
        result['attn_output_gate'] = 1 if attnGate else 0

        layerBits = 0
        for i, lt in enumerate(tc['layer_types']):
            if lt == 'full_attention':
                layerBits |= (1 << i)
        # Convert to signed int32 for the binary header
        if layerBits >= 0x80000000:
            layerBits -= 0x100000000
        result['layer_type_bits'] = layerBits

        return result

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

    nExperts = config.get('num_experts')
    nActiveExperts = config.get('num_experts_per_tok')
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

    headDim = config.get('head_dim')
    if (headDim is not None):
        result['head_dim'] = headDim

    rmsNormEps = config.get('rms_norm_eps')
    if (rmsNormEps is not None):
        result['norm_epsilon'] = parseRmsNormEpsilon(rmsNormEps)

    moeHiddenDim = config.get('moe_intermediate_size')
    if (moeHiddenDim is not None):
        result['moe_hidden_dim'] = int(moeHiddenDim)
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
    tmpFileName = outputFileName + '.tmp'

    print(f'Output file: {outputFileName}')

    config = loadConfig(sourceFolderPath, weightsFloatType)

    try:
        with open(tmpFileName, 'wb') as outputFile:
            writeHeader(outputFile, config)
            processor = Processor(config)
            processor.write(outputFile)
        os.replace(tmpFileName, outputFileName)
        print(f'✅ {outputFileName} created successfully')
    except KeyboardInterrupt:
        if os.path.exists(tmpFileName):
            try:
                os.remove(tmpFileName)
            except OSError:
                pass
        print('\n🛑 Conversion aborted by user (Ctrl+C). Partial tmp removed.', flush=True)
        raise
    except Exception as e:
        failed_name = f'{outputFileName}.failed-{int(datetime.datetime.now().timestamp() * 1_000_000)}'
        if os.path.exists(tmpFileName):
            try:
                os.replace(tmpFileName, failed_name)
                partial_size = os.path.getsize(failed_name)
                print(f'\n❌ Conversion aborted: {type(e).__name__}: {e}', flush=True)
                print(f'   Source safetensors file loaded: {[os.path.basename(f) for f in config["files"]]}', flush=True)
                print(f'   Partial output preserved as: {failed_name} ({partial_size:,} bytes, {partial_size/1024**3:.2f} GiB)', flush=True)
            except OSError as move_err:
                print(f'\n❌ Conversion aborted: {type(e).__name__}: {e}', flush=True)
                print(f'   Could not preserve partial output: {move_err}', flush=True)
        else:
            print(f'\n❌ Conversion aborted before any output was written: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()
        sys.exit(1)