import os
import sys
import requests

# ['model-url', 'tokenizer-url', 'weights-float-type', 'buffer-float-type', 'model-type']
MODELS = {
    'llama3_8b_instruct_q40': [
        'https://huggingface.co/b4rtaz/Llama-3-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_lama3_instruct_q40.m?download=true',
        'https://huggingface.co/b4rtaz/Llama-3-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama3.t?download=true',
        'q40', 'q80', 'chat'
    ]
}

ALIASES = {
    'llama3_instruct': 'llama3_8b_instruct_q40'
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

    model = MODELS[modelName]
    (modelPath, tokenizerPath) = download(modelName, model)
    if (model[4] == 'chat'):
        command = './dllama chat'
    else:
        command = './dllama inference --steps 64 --prompt "Hello world"'
    command += f' --model {modelPath} --tokenizer {tokenizerPath} --buffer-float-type {model[3]} --nthreads 4'

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
