import struct
import torch
import time
import numpy as np

def isFloatTypeSupported(type):
    return type in ['f16', 'f32', 'q40']

def writeQuantizedQ40Tensor(file, x):
    t0 = time.time()
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
    t1 = time.time()
    print(f'Quantized tensor to {nBytes} bytes in {t1 - t0:.2f} s')

def writeF32Tensor(file, d):
    chunkSize = 10000
    for i in range(0, len(d), chunkSize):
        chunk = d[i:i+chunkSize].to(torch.float32).numpy().astype(np.float32)
        b = struct.pack(f'{len(chunk)}f', *chunk)
        file.write(b)

def writeTensor(file, tensor, floatType):
    d = tensor.detach().cpu().view(-1)
    if (floatType == 'f16'):
        d = d.to(torch.float16).numpy().astype(np.float16)
        b = struct.pack(f'{len(d)}e', *d)
        file.write(b)
    elif (floatType == 'f32'):
        writeF32Tensor(file, d)
    elif (floatType == 'q40'):
        writeQuantizedQ40Tensor(file, d)
    else:
        raise Exception('Unknown float type')

def writeHeader(file, params):
    header = struct.pack('iiiiiiiiii',
        params['arch_type'],
        params['dim'],
        params['hidden_dim'],
        params['n_layers'],
        params['n_heads'],
        params['n_kv_heads'],
        params['n_experts'],
        params['n_active_experts'],
        params['vocab_size'],
        params['max_seq_len'])
    file.write(header)
    print(params)
