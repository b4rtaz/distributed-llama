import os
import sys
import requests

def parts(length):
    result = []
    for i in range(length):
        a = chr(97 + (i // 26))
        b = chr(97 + (i % 26))
        result.append(a + b)
    return result

# [['model-url-0', 'model-url-1', ...], 'tokenizer-url', 'weights-float-type', 'buffer-float-type', 'model-type']
MODELS = {
    'tinyllama_1_1b_3t_q40': [
        ['https://huggingface.co/b4rtaz/TinyLlama-1.1B-3T-Distributed-Llama/resolve/main/dllama_model_tinylama_1.1b_3t_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/TinyLlama-1.1B-3T-Distributed-Llama/resolve/main/dllama_tokenizer_tinylama_1.1b_3t.t?download=true',
        'q40', 'q80', 'base'
    ],
    'llama3_8b_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3-8B-Q40-Distributed-Llama/resolve/main/dllama_model_meta-llama-3-8b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3-8B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_llama3.t?download=true',
        'q40', 'q80', 'base'
    ],
    'llama3_8b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_lama3_instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama3.t?download=true',
        'q40', 'q80', 'chat'
    ],
    'llama3_1_8b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3_1-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama3.1_instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3_1-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama_3_1.t?download=true',
        'q40', 'q80', 'chat'
    ],
    'llama3_1_405b_instruct_q40': [
        list(map(lambda suffix : f'https://huggingface.co/b4rtaz/Llama-3_1-405B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama31_405b_q40_{suffix}?download=true', parts(56))),
        'https://huggingface.co/b4rtaz/Llama-3_1-405B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama_3_1.t?download=true',
        'q40', 'q80', 'chat'
    ],
}

def downloadFile(urls: str, path: str):
    if (os.path.isfile(path)):
        fileName = os.path.basename(path)
        result = input(f'❓ {fileName} already exists, do you want to download again? ("Y" if yes): ')
        if (result.upper() != 'Y'):
            return

    lastSizeMb = 0
    with open(path, 'wb') as file:
        for url in urls:
            print(f'📄 {url}')
            response = requests.get(url, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=4096):
                file.write(chunk)
                sizeMb = file.tell() // (1024 * 1024)
                if (sizeMb != lastSizeMb):
                    sys.stdout.write("\rDownloaded %i MB" % sizeMb)
                    lastSizeMb = sizeMb
            sys.stdout.write('\n')
    sys.stdout.write(' ✅\n')

def download(modelName: str, model: list):
    dirPath = os.path.join('models', modelName)
    print(f'📀 Downloading {modelName} to {dirPath}...')
    os.makedirs(dirPath, exist_ok=True)
    modelUrls = model[0]
    tokenizerUrl = model[1]
    modelPath = os.path.join(dirPath, f'dllama_model_{modelName}.m')
    tokenizerPath = os.path.join(dirPath, f'dllama_tokenizer_{modelName}.t')
    downloadFile(modelUrls, modelPath)
    downloadFile([tokenizerUrl], tokenizerPath)
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
