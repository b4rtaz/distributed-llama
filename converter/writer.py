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
    """Writes Q40-quantized tensor, streaming in 256-block chunks.

    Bounds peak RAM at ~80 KB (chunk working set) regardless of input size.
    Per-block output: [float16 delta (2B), 16 nibble-packed uint8 values (16B)] = 18 bytes.
    Byte layout matches the previous implementation on the wire.
    """
    blockSize = 32
    blockHalfSize = blockSize // 2

    # Flatten to 1D view (no copy if tensor is already contiguous F32)
    x_flat = x.detach().to(torch.float32).reshape(-1)
    n = x_flat.numel()
    assert (n % blockSize == 0), f'Q40 quantization requires rows % 32 == 0 (got {n})'
    nBlocks = n // blockSize

    CHUNK_BLOCKS = 256  # 256 * 32 = 8192 elements (~32 KB F32) per chunk

    nBytes = 0
    for i in range(0, nBlocks, CHUNK_BLOCKS):
        chunkEnd = i + CHUNK_BLOCKS if i + CHUNK_BLOCKS < nBlocks else nBlocks
        chunkN = chunkEnd - i
        # Per-chunk working set: ~80 KB F32, ~30 KB uint8 + small temps
        chunk_np = x_flat[i * blockSize : chunkEnd * blockSize].cpu().numpy().astype(np.float32, copy=False)
        blockView = chunk_np.reshape(chunkN, blockSize)
        gmax = blockView.max(axis=1)
        gmin = blockView.min(axis=1)
        deltas = np.divide(np.where(-gmin > gmax, gmin, gmax), -8)
        deltas16 = deltas.astype(np.float16)
        ids = np.where(deltas != 0, 1.0 / deltas, 0)
        # uint8 (not int!) to keep the clipped-quantized array in 1 byte/element
        groups8 = np.clip(blockView * ids[:, np.newaxis] + 8.5, 0, 15).astype(np.uint8)
        gLow = groups8[:, :blockHalfSize]
        gHigh = (groups8[:, blockHalfSize:] << 4) & 0xF0
        gCombined = (gLow | gHigh).astype(np.uint8)

        # Interleave via numpy view: each row is [delta16(2B) | nibble-packed(16B)]
        buf = np.empty((chunkN, 2 + blockHalfSize), dtype=np.uint8)
        buf[:, 0:2] = deltas16.view(np.uint8).reshape(chunkN, 2)
        buf[:, 2:2 + blockHalfSize] = gCombined
        chunkBytes = buf.tobytes()
        file.write(chunkBytes)
        nBytes += len(chunkBytes)
        # Free chunk intermediates so they don't accumulate across iterations
        del chunk_np, blockView, gmax, gmin, deltas, deltas16, ids, groups8, gLow, gHigh, gCombined, buf, chunkBytes
    return nBytes

def writeQuantizedQ80Tensor(file, x):
    """Writes Q80-quantized tensor, streaming in 256-block chunks.

    Bounds peak RAM at ~80 KB regardless of input size.
    Per-block output: [float16 delta (2B), 32 int8 values (32B)] = 34 bytes.
    """
    blockSize = 32

    x_flat = x.detach().to(torch.float32).reshape(-1)
    n = x_flat.numel()
    assert (n % blockSize == 0), f'Q80 quantization requires rows % 32 == 0 (got {n})'
    nBlocks = n // blockSize

    CHUNK_BLOCKS = 256

    nBytes = 0
    for i in range(0, nBlocks, CHUNK_BLOCKS):
        chunkEnd = i + CHUNK_BLOCKS if i + CHUNK_BLOCKS < nBlocks else nBlocks
        chunkN = chunkEnd - i
        chunk_np = x_flat[i * blockSize : chunkEnd * blockSize].cpu().numpy().astype(np.float32, copy=False)
        blockView = chunk_np.reshape(chunkN, blockSize)
        gmin = blockView.min(axis=1)
        gmax = blockView.max(axis=1)
        gabsMax = np.where(-gmin > gmax, -gmin, gmax)
        deltas = gabsMax / ((1 << 7) - 1)
        deltas16 = deltas.astype(np.float16)
        ids = np.where(deltas != 0, 1.0 / deltas, 0)
        groups8 = np.round(blockView * ids[:, np.newaxis]).astype(np.int8)

        # Interleave: each row is [delta16(2B) | 32 int8 values(32B)]
        buf = np.empty((chunkN, 2 + blockSize), dtype=np.uint8)
        buf[:, 0:2] = deltas16.view(np.uint8).reshape(chunkN, 2)
        buf[:, 2:2 + blockSize] = groups8.view(np.uint8).reshape(chunkN, blockSize)
        chunkBytes = buf.tobytes()
        file.write(chunkBytes)
        nBytes += len(chunkBytes)
        del chunk_np, blockView, gmin, gmax, gabsMax, deltas, deltas16, ids, groups8, buf, chunkBytes
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
        'head_dim': 19,
        'norm_epsilon': 20,
        'moe_hidden_dim': 21,
        'partial_rotary_factor': 22,
        'attn_output_gate': 23,
        'layer_type_bits': 24,
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
