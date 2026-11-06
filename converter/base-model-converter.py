import gc
import json
import os
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

class FloatType:
    F32 = 0
    F16 = 1
    Q40 = 2
    Q80 = 3

floatTypeMap = {
    'f32': FloatType.F32,
    'f16': FloatType.F16,
    'q40': FloatType.Q40,
    'q80': FloatType.Q80,
}
floatTypeNames = list(floatTypeMap.keys())

def parseFloatType(type):
    floatType = floatTypeMap.get(type)
    if floatType is not None:
        return floatType
    raise Exception(f'{type} is not supported')

def strFloatType(type):
    return floatTypeNames[type]

def writeQuantizedQ40Tensor(file, x):
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
    groups = np.clip(groups, 0, 15).astype(int)

    gLow = groups[:, :blockHalfSize] & 0xF
    gHigh = (groups[:, blockHalfSize:] & 0xF) << 4
    gCombined = gLow | gHigh

    nBytes = 0
    for groupIndex in range(0, len(groups)):
        delta16 = deltas16[groupIndex]
        buffer = struct.pack(f'e{blockHalfSize}B', delta16, *gCombined[groupIndex])
        file.write(buffer)
        nBytes += len(buffer)
    return nBytes

def writeQuantizedQ80Tensor(file, x):
    x = x.to(torch.float32).numpy().astype(np.float32)
    blockSize = 32
    assert(x.shape[0] % blockSize == 0)
    groups = x.reshape(-1, blockSize)
    gmax = np.max(groups, axis=1)
    gmin = np.min(groups, axis=1)
    gabsMax = np.where(-gmin > gmax, -gmin, gmax)
    deltas = gabsMax / ((1 << 7) - 1)
    deltas16 = deltas.astype(np.float16)
    ids = np.where(deltas != 0, 1.0 / deltas, 0)
    groups = groups * ids[:, np.newaxis]
    groups8 = np.round(groups).astype(np.int8)

    nBytes = 0
    for groupIndex in range(0, len(groups)):
        buffer = struct.pack(f'e{blockSize}b', deltas16[groupIndex], *groups8[groupIndex])
        file.write(buffer)
        nBytes += len(buffer)
    return nBytes

def writeF32Tensor(file, d):
    chunkSize = 10000
    nBytes = 0
    for i in range(0, len(d), chunkSize):
        chunk = d[i:i+chunkSize].to(torch.float32).numpy().astype(np.float32)
        b = struct.pack(f'{len(chunk)}f', *chunk)
        nBytes += len(b)
        file.write(b)
    return nBytes

def writeF16Tensor(file, d):
    d = d.to(torch.float16).numpy().astype(np.float16)
    b = struct.pack(f'{len(d)}e', *d)
    file.write(b)
    return len(b)

def writeTensor(file, tensor, floatType):
    d = tensor.detach().cpu().view(-1)
    t0 = time.time()
    nBytes = 0
    if (floatType == FloatType.F16):
        nBytes = writeF16Tensor(file, d)
    elif (floatType == FloatType.F32):
        nBytes = writeF32Tensor(file, d)
    elif (floatType == FloatType.Q40):
        nBytes = writeQuantizedQ40Tensor(file, d)
    elif (floatType == FloatType.Q80):
        nBytes = writeQuantizedQ80Tensor(file, d)
    else:
        raise Exception(f'Unknown float type')
    t1 = time.time()
    print(f'Saved {strFloatType(floatType)} tensor in {t1 - t0:.2f}s, {nBytes} bytes')

def writeHeader(file, magic, values, headerKeys):
    keyValues = []
    for key, value in values.items():
        keyId = headerKeys.get(key)
        if keyId is None:
            print(f'Warning: Unknown header key: {key}')
            continue
        keyValues.append((keyId, value))
        print(f'🎓 {key}: {value}')

    data = b''.join(struct.pack('ii', keyId, value) for keyId, value in keyValues)
    file.write(struct.pack('ii', magic, 2 * struct.calcsize('i') + len(data)))
    file.write(data)
    print()


def parseRmsNormEpsilon(epsilon: float):
    if epsilon == 1e-5:
        return 5
    if epsilon == 1e-6:
        return 6
    raise Exception(f'Unsupported RMS norm epsilon: {epsilon}')


def parseHiddenAct(act: str):
    hiddenAct = {'gelu': 0, 'silu': 1}.get(act)
    if hiddenAct is None:
        raise Exception(f'Unsupported hidden activation: {act}')
    return hiddenAct


def parseRopeType(ropeType: str):
    result = {'llama3': 2}.get(ropeType)
    if result is None:
        raise Exception(f'Unsupported rope type: {ropeType}')
    return result


def loadWeightMap(folderPath: str):
    indexPath = os.path.join(folderPath, 'model.safetensors.index.json')
    if os.path.exists(indexPath):
        with open(indexPath) as file:
            index = json.load(file)
        return index['weight_map']

    weightMap = {}
    files = sorted(
        fileName for fileName in os.listdir(folderPath)
        if fileName.endswith('.safetensors') and not fileName.startswith('.')
    )
    if not files:
        raise Exception('No safetensors model files found')
    for fileName in files:
        with safe_open(os.path.join(folderPath, fileName), framework='pt', device='cpu') as model:
            for key in model.keys():
                if key in weightMap:
                    raise Exception(f'Tensor {key} occurs in more than one shard')
                weightMap[key] = fileName
    return weightMap


class BaseModelPlan:
    headerMagic = None
    headerKeys = None

    def __init__(self, folderPath, sourceConfig, weightsFloatType, weightMap):
        self.folderPath = folderPath
        self.sourceConfig = sourceConfig
        self.weightsFloatType = weightsFloatType
        self.weightMap = weightMap
        self.header = {}
        self.tensors = []

    def addTensor(
        self, outputName, floatType, sourceNames, sourceShape,
        outputShape=None, transform=None
    ):
        if isinstance(sourceNames, str):
            sourceNames = [sourceNames]
        sourceName = next((name for name in sourceNames if name in self.weightMap), None)
        if sourceName is None:
            raise Exception(f"None of the source tensors were found: {', '.join(sourceNames)}")
        self.tensors.append({
            'output_name': outputName,
            'float_type': floatType,
            'source_name': sourceName,
            'source_shape': tuple(sourceShape),
            'output_shape': tuple(outputShape if outputShape is not None else sourceShape),
            'transform': transform,
        })


class ModelConverter:
    def __init__(self, plan):
        self.plan = plan
        self.currentShardName = None
        self.currentShard = None
        self.currentTensorName = None
        self.currentTensor = None

    def __unload(self):
        self.currentTensor = None
        self.currentTensorName = None
        self.currentShard = None
        self.currentShardName = None
        gc.collect()

    def __loadTensor(self, tensorName):
        if self.currentTensorName == tensorName:
            return self.currentTensor
        self.currentTensor = None
        self.currentTensorName = None

        shardName = self.plan.weightMap[tensorName]
        if self.currentShardName != shardName:
            self.currentShard = None
            gc.collect()
            shardPath = os.path.join(self.plan.folderPath, shardName)
            print(f'💿 Loading file {shardName}...')
            self.currentShard = safe_open(shardPath, framework='pt', device='cpu')
            self.currentShardName = shardName

        self.currentTensor = self.currentShard.get_tensor(tensorName)
        self.currentTensorName = tensorName
        return self.currentTensor

    def validate(self):
        requiredByShard = {}
        for item in self.plan.tensors:
            tensorName = item['source_name']
            shardName = self.plan.weightMap[tensorName]
            requiredByShard.setdefault(shardName, {})[tensorName] = item['source_shape']

        for shardName, tensors in sorted(requiredByShard.items()):
            shardPath = os.path.join(self.plan.folderPath, shardName)
            with safe_open(shardPath, framework='pt', device='cpu') as model:
                for tensorName, expectedShape in tensors.items():
                    actualShape = tuple(model.get_slice(tensorName).get_shape())
                    if actualShape != expectedShape:
                        raise Exception(
                            f'Invalid shape for {tensorName}: expected {expectedShape}, received {actualShape}'
                        )
        print(f'✅ Validated {len(self.plan.tensors)} output tensors')

    def write(self, outputFile):
        for item in self.plan.tensors:
            tensor = self.__loadTensor(item['source_name'])
            transform = item['transform']
            if transform is not None:
                tensor = transform(tensor)
            if tuple(tensor.shape) != item['output_shape']:
                raise Exception(
                    f"Invalid transformed shape for {item['output_name']}: "
                    f"expected {item['output_shape']}, received {tuple(tensor.shape)}"
                )
            print(
                f"🔶 Writing {item['output_name']} from {item['source_name']} "
                f"{tuple(tensor.shape)}..."
            )
            writeTensor(outputFile, tensor, item['float_type'])
        self.__unload()


def runConverter(planType, description, exampleName):
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print(
            f'Usage: python {os.path.basename(sys.argv[0])} '
            '<sourceFolderPath> <weightsFloatType> <name> [--dry-run]'
        )
        print()
        print('Options:')
        print(f'  <sourceFolderPath> The folder containing the {description} config and safetensors files')
        print('  <weightsFloatType> The matrix weight type (for example "q40")')
        print(f'  <name>             Output model name (for example "{exampleName}")')
        print('  --dry-run          Validate configuration, tensor names, and shapes without writing')
        exit(1)

    folderPath = sys.argv[1]
    weightsFloatTypeName = sys.argv[2]
    weightsFloatType = parseFloatType(weightsFloatTypeName)
    name = sys.argv[3]
    dryRun = len(sys.argv) == 5 and sys.argv[4] == '--dry-run'
    if len(sys.argv) == 5 and not dryRun:
        raise Exception(f'Unsupported option: {sys.argv[4]}')

    with open(os.path.join(folderPath, 'config.json')) as file:
        sourceConfig = json.load(file)
    weightMap = loadWeightMap(folderPath)
    plan = planType(folderPath, sourceConfig, weightsFloatType, weightMap)
    converter = ModelConverter(plan)
    converter.validate()
    if dryRun:
        return

    outputFileName = f'dllama_model_{name}_{weightsFloatTypeName}.m'
    print(f'Output file: {outputFileName}')
    with open(outputFileName, 'wb') as outputFile:
        writeHeader(outputFile, plan.headerMagic, plan.header, plan.headerKeys)
        converter.write(outputFile)
    print(f'✅ {outputFileName} created successfully')
