import os
import sys
import requests

MODELS = {
    'llama3_8b_q40': [
        'https://huggingface.co/b4rtaz/llama-3-8b-distributed-llama/resolve/main/dllama_meta-llama-3-8b_q40.bin?download=true',
        'https://huggingface.co/b4rtaz/llama-3-8b-distributed-llama/resolve/main/dllama_meta-llama3-tokenizer.t?download=true',
    ],
    'llama3_8b_instruct_q40': [
        'https://huggingface.co/Azamorn/Meta-Llama-3-8B-Instruct-Distributed/resolve/main/dllama_original_q40.bin?download=true',
        'https://huggingface.co/Azamorn/Meta-Llama-3-8B-Instruct-Distributed/resolve/main/dllama-llama3-tokenizer.t?download=true',
    ],
    'tinylama_1.1b_3t_q40': [
        'https://huggingface.co/b4rtaz/tinyllama-1.1b-1431k-3t-distributed-llama/resolve/main/dllama_model_tinylama_1.1b_3t_q40.m?download=true',
        'https://huggingface.co/b4rtaz/tinyllama-1.1b-1431k-3t-distributed-llama/resolve/main/dllama_tokenizer_tinylama_1.1b_3t_q40.t?download=true'
    ]
}

ALIASES = {
    'llama3': 'llama3_8b_q40',
    'llama3_8b': 'llama3_8b_q40',
    'llama3_instruct': 'llama3_8b_instruct_q40',
    'llama3_8b_instruct': 'llama3_8b_instruct_q40',
    'tinylama': 'tinylama_1.1b_3t_q40'
}

def downloadFile(url: str, path: str):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    print(f'📄 {url}')
    lastSize = 0
    with open(path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=4096):
            file.write(chunk)
            size = file.tell() // 1024
            if (size - lastSize >= 8192):
                sys.stdout.write("\rDownloaded %i kB" % size)
                lastSize = size
    sys.stdout.write(' ✅\n')

def download(modelName: str, model: list):
    dirPath = os.path.join('models', modelName)
    print(f'📀 Downloading {modelName} to {dirPath}...')
    os.makedirs(dirPath, exist_ok=True)
    modelUrl = model[0]
    tokenizerUrl = model[1]
    modelPath = os.path.join(dirPath, f'dllama_model_{modelName}.m')
    tokenizerPath = os.path.join(dirPath, f'dllama_tokenizer_{modelName}.t')
    downloadFile(modelUrl, modelPath)
    downloadFile(tokenizerUrl, tokenizerPath)
    print('📀 All files are downloaded')
    return (modelPath, tokenizerPath)

def writeRunFile(modelName: str, command: str):
    filePath = f'run_{modelName}.sh'
    with open(filePath, 'w') as file:
        file.write('#!/bin/sh\n')
        file.write('\n')
        file.write(f'{command}\n')
    return filePath

def printUsage():
    print('Usage: python download-model.py <model>')
    print('Available models:')
    for model in MODELS:
        print(f'  {model}')

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        printUsage()
        exit(1)

    os.chdir(os.path.dirname(__file__))

    modelName = sys.argv[1].replace('-', '_')
    if modelName in ALIASES:
        modelName = ALIASES[modelName]
    if modelName not in MODELS:
        print(f'Model is not supported: {modelName}')
        exit(1)

    (modelPath, tokenizerPath) = download(modelName, MODELS[modelName])
    command = f'./dllama inference --model {modelPath} --tokenizer {tokenizerPath} --weights-float-type q40 --buffer-float-type q80 --nthreads 4 --steps 64 --prompt "Hello world"'

    print('To run Distributed Llama you need to execute:')
    print('--- copy start ---')
    print()
    print(command)
    print()
    print('--- copy end -----')

    runFilePath = writeRunFile(modelName, command)
    print(f'🌻 Created {runFilePath} script to easy run')

    result = input('❓ Do you want to run Distributed Llama? ("Y" if yes): ')
    if (result.upper() == 'Y'):
        if (not os.path.isfile('dllama')):
            os.system('make dllama')
        os.system(command)
