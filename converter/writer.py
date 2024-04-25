import struct
import torch
import time
import numpy as np

def isFloatTypeSupported(type):
    return type in ['f16', 'f32', 'q40', 'q80']

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
    groups = np.where(groups < 15, groups, 15)

    nBytes = 0
    block = [0] * blockHalfSize
    for groupIndex in range(0, len(groups)):
        group = groups[groupIndex]
        delta16 = deltas16[groupIndex]

        for i in range(0, blockHalfSize):
            x0 = int(group[i])
            x1 = int(group[i + blockHalfSize])
            block[i] = (x0 & 0xF) | ((x1 & 0xF) << 4)

        buffer = struct.pack(f'e{blockHalfSize}B', delta16, *block)
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
    if (floatType == 'f16'):
        nBytes = writeF16Tensor(file, d)
    elif (floatType == 'f32'):
        nBytes = writeF32Tensor(file, d)
    elif (floatType == 'q40'):
        nBytes = writeQuantizedQ40Tensor(file, d)
    elif (floatType == 'q80'):
        nBytes = writeQuantizedQ80Tensor(file, d)
    else:
        raise Exception('Unknown float type')
    t1 = time.time()
    print(f'Saved {floatType} tensor in {t1 - t0:.2f}s, {nBytes} bytes')

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
    }
    header = struct.pack('i', 0xA00ABCD)

    data = b''
    for key in params:
        if key in headerKeys:
            data += struct.pack('ii', headerKeys[key], params[key])
        else:
            print(f'Unknown header key: {key}')

    header += struct.pack('i', len(header) * 2 + len(data))
    file.write(header)
    file.write(data)
    print(params)
