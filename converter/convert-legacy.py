import sys
import struct
from pathlib import Path

F32_BYTES = 4

def usage():
    print('Usage: python convert-legacy.py <modelPath> <sharedWeights>')
    exit(1)

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        usage()

    modelPath = sys.argv[1]
    modelName = Path(modelPath).stem
    sharedWeights = sys.argv[2] == 'true'
    print(f'Model name: {modelName}')

    sourceFile = open(modelPath, 'rb')
    headerBytes = sourceFile.read(7 * F32_BYTES)
    (dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, maxSeqLen) = struct.unpack('7i', headerBytes)

    kvDim = (dim * nKvHeads) // nHeads
    headSize = dim // nHeads

    embeddingBytes = vocabSize * dim * F32_BYTES
    rmsAttBytes = dim * F32_BYTES
    rmsFfnBytes = dim * F32_BYTES
    qBytes = dim * dim * F32_BYTES
    kBytes = dim * kvDim * F32_BYTES
    vBytes = dim * kvDim * F32_BYTES
    woBytes = dim * dim * F32_BYTES
    w1Bytes = dim * hiddenDim * F32_BYTES
    w2Bytes = hiddenDim * dim * F32_BYTES
    w3Bytes = dim * hiddenDim * F32_BYTES
    rmsFinalBytes = dim * F32_BYTES
    freqCisRealBytes = maxSeqLen * headSize / 2 * F32_BYTES
    freqCisImagBytes = maxSeqLen * headSize / 2 * F32_BYTES
    wclsBytes = vocabSize * dim * F32_BYTES

    targetPath = 'dllama_' + modelName + '.bin'
    targetFile = open(targetPath, 'wb')
    targetFile.write(headerBytes)

    embeddingData = sourceFile.read(embeddingBytes)
    targetFile.write(embeddingData)

    rmsAtts = [sourceFile.read(rmsAttBytes) for _ in range(nLayers)]
    qs = [sourceFile.read(qBytes) for _ in range(nLayers)]
    ks = [sourceFile.read(kBytes) for _ in range(nLayers)]
    vs = [sourceFile.read(vBytes) for _ in range(nLayers)]
    wos = [sourceFile.read(woBytes) for _ in range(nLayers)]
    rmsFfns = [sourceFile.read(rmsFfnBytes) for _ in range(nLayers)]
    w1s = [sourceFile.read(w1Bytes) for _ in range(nLayers)]
    w2s = [sourceFile.read(w2Bytes) for _ in range(nLayers)]
    w3s = [sourceFile.read(w3Bytes) for _ in range(nLayers)]

    for i in range(nLayers):
        targetFile.write(rmsAtts[i])
        targetFile.write(rmsFfns[i])
        targetFile.write(qs[i])
        targetFile.write(ks[i])
        targetFile.write(vs[i])
        targetFile.write(wos[i])
        targetFile.write(w1s[i])
        targetFile.write(w2s[i])
        targetFile.write(w3s[i])

    targetFile.write(sourceFile.read(rmsFinalBytes))

    targetFile.seek(int(freqCisRealBytes), 1)
    targetFile.seek(int(freqCisImagBytes), 1)
    if (sharedWeights):
        targetFile.write(embeddingData)
    else:
        targetFile.write(sourceFile.read(wclsBytes))

    bytes = targetFile.tell()
    targetFile.close()
    sourceFile.close()

    print('Done!')
