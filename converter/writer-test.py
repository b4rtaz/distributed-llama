import sys
import time
import torch
from writer import writeQuantizedQ40Tensor

TEMP_FILE_NAME = 'writer-test.temp'

def readBase64FromFile(path):
    with open(path, 'rb') as file:
        return file.read().hex()

def testWriteQuantizedQ40Tensor():
    EXPECTED_OUTPUT = '7e346345a692b89665b2c5790537876e598aaa366d988876a898b8d788a98868ce660c66f6b3a88cba5ce9a871987ba9cc5bcaaa760c1eb556a4455b747b6b9504968828ef2a8d7c1db5c6be3764799e66db6d8e76463126a30e4333cad7a4f645947c6cf97f9de086d468c8d535a6ba7dc799d3d0c657bab6799468cad8bb349eb7d7635c7c798998696bb38e4085a9eb34444ba96a7f8ba7b2b42d746a96cf9660aeb4499d8708ad5c7b9a7558947645f3bbb6b0346a656887ad9a86059baac5c596ab781c703569bb8a4356a4bd58cb78736ba09759bb0e34a6274e827b957d7a67dfa86846955660d234b6d9d78a378094a8a8708a7a774ae92f8a36b8c999a9b77a7d958a69747c807963941235379886d69a7a8767b3a6a4ac71999760'

    torch.manual_seed(seed=1)
    tensor = torch.randn(32, 16)

    with open(TEMP_FILE_NAME, 'wb') as file:
        writeQuantizedQ40Tensor(file, tensor)

    contentBase64 = readBase64FromFile(TEMP_FILE_NAME)
    assert contentBase64 == EXPECTED_OUTPUT, f'Received: {contentBase64}'
    print('✅ writeQuantizedQ40Tensor')

def runWriteQuantizedQ40TensorBenchmark():
    tensor = torch.randn(8192, 4096)
    t0 = time.time()
    with open(TEMP_FILE_NAME, 'wb') as file:
        writeQuantizedQ40Tensor(file, tensor)
    t1 = time.time()
    print(f'🕐 writeQuantizedQ40Tensor: {t1 - t0:.4f}s')

if __name__ == '__main__':
    testWriteQuantizedQ40Tensor()
    runWriteQuantizedQ40TensorBenchmark()
