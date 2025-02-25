import struct
import torch
import time
import numpy as np

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

def writeHeader(file, params):
    headerKeys = {
        'version': 0,
        'arch_type': 1,
        'dim': 2,
        'hidden_dim': 3,
        'n_layers': 4,
        'n_heads': 5,
        'n_kv_heads': 6,
        'n_experts': 7,
        'n_active_experts': 8,
        'vocab_size': 9,
        'max_seq_len': 10,
        'hidden_act': 11,
        'rope_theta': 12,
        'weights_float_type': 13,
        'rope_scaling_factor': 14,
        'rope_scaling_low_freq_factor': 15,
        'rope_scaling_high_freq_factory': 16,
        'rope_scaling_orig_max_seq_len': 17,
        'rope_type': 18,
    }
    header = struct.pack('i', 0xA00ABCD)

    data = b''
    for key in params:
        if key in headerKeys:
            data += struct.pack('ii', headerKeys[key], params[key])
        else:
            print(f'Warning: Unknown header key: {key}')

    header += struct.pack('i', len(header) * 2 + len(data))
    file.write(header)
    file.write(data)
    for key in params:
        print(f'🎓 {key}: {params[key]}')
    print()
