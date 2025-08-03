import os
import sys
import time
import multiprocessing
from urllib.request import urlopen

def parts(length):
    result = []
    for i in range(length):
        a = chr(97 + (i // 26))
        b = chr(97 + (i % 26))
        result.append(a + b)
    return result

# [['model-url-0', 'model-url-1', ...], 'tokenizer-url', 'weights-float-type', 'buffer-float-type', 'model-type']
MODELS = {
    'llama3_1_8b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3_1-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama3.1_instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3_1-8B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama_3_1.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_1_405b_instruct_q40': [
        list(map(lambda suffix : f'https://huggingface.co/b4rtaz/Llama-3_1-405B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama31_405b_q40_{suffix}?download=true', parts(56))),
        'https://huggingface.co/b4rtaz/Llama-3_1-405B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama_3_1.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_2_1b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3_2-1B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama3.2-1b-instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3_2-1B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama3_2.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_2_3b_instruct_q40': [
        ['https://huggingface.co/b4rtaz/Llama-3_2-3B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama3.2-3b-instruct_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Llama-3_2-3B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama3_2.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'llama3_3_70b_instruct_q40': [
        list(map(lambda suffix : f'https://huggingface.co/b4rtaz/Llama-3_3-70B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_model_llama-3.3-70b_q40{suffix}?download=true', parts(11))),
        'https://huggingface.co/b4rtaz/Llama-3_3-70B-Q40-Instruct-Distributed-Llama/resolve/main/dllama_tokenizer_llama-3.3-70b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'deepseek_r1_distill_llama_8b_q40': [
        ['https://huggingface.co/b4rtaz/DeepSeek-R1-Distill-Llama-8B-Distributed-Llama/resolve/main/dllama_model_deepseek-r1-distill-llama-8b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/DeepSeek-R1-Distill-Llama-8B-Distributed-Llama/resolve/main/dllama_tokenizer_deepseek-r1-distill-llama-8b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_0.6b_q40': [
        ['https://huggingface.co/b4rtaz/Qwen3-0.6B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_0.6b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Qwen3-0.6B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_0.6b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_1.7b_q40': [
        ['https://huggingface.co/b4rtaz/Qwen3-1.7B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_1.7b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Qwen3-1.7B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_1.7b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
    'qwen3_8b_q40': [
        ['https://huggingface.co/b4rtaz/Qwen3-8B-Q40-Distributed-Llama/resolve/main/dllama_model_qwen3_8b_q40.m?download=true'],
        'https://huggingface.co/b4rtaz/Qwen3-8B-Q40-Distributed-Llama/resolve/main/dllama_tokenizer_qwen3_8b.t?download=true',
        'q40', 'q80', 'chat', '--max-seq-len 4096'
    ],
}

def confirm(message: str):
    result = input(f'❓ {message} ("Y" if yes): ').upper()
    return result == 'Y' or result == 'YES'

def downloadFile(urls, path: str):
    if os.path.isfile(path):
        fileName = os.path.basename(path)
        if not confirm(f'{fileName} already exists, do you want to download again?'):
            return

    lastSizeMb = 0
    with open(path, 'wb') as file:
        for url in urls:
            startPosition = file.tell()
            success = False
            for attempt in range(8):
                print(f'📄 {url} (attempt: {attempt})')
                try:
                    with urlopen(url) as response:
                        while True:
                            chunk = response.read(4096)
                            if not chunk:
                                break
                            file.write(chunk)
                            sizeMb = file.tell() // (1024 * 1024)
                            if sizeMb != lastSizeMb:
                                sys.stdout.write("\rDownloaded %i MB" % sizeMb)
                                lastSizeMb = sizeMb
                    sys.stdout.write('\n')
                    success = True
                    break
                except Exception as e:
                    print(f'\n❌ Error downloading {url}: {e}')
                file.seek(startPosition)
                file.truncate()
                time.sleep(1 * attempt)
            if not success:
                raise Exception(f'Failed to download {url}')
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
    print()
    print('Options:')
    print('  <model> The name of the model to download')
    print('  --run Run the model after download')
    print()
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
    runAfterDownload = sys.argv.count('--run') > 0

    model = MODELS[modelName]
    (modelPath, tokenizerPath) = download(modelName, model)

    nThreads = multiprocessing.cpu_count()
    if (model[4] == 'chat'):
        command = './dllama chat'
    else:
        command = './dllama inference --steps 64 --prompt "Hello world"'
    command += f' --model {modelPath} --tokenizer {tokenizerPath} --buffer-float-type {model[3]} --nthreads {nThreads}'
    if (len(model) > 5):
        command += f' {model[5]}'

    print('To run Distributed Llama you need to execute:')
    print('--- copy start ---')
    print()
    print('\033[96m' + command + '\033[0m')
    print()
    print('--- copy end -----')

    runFilePath = writeRunFile(modelName, command)
    print(f'🌻 Created {runFilePath} script to easy run')

    if (not runAfterDownload):
        runAfterDownload = confirm('Do you want to run Distributed Llama?')
    if (runAfterDownload):
        if (not os.path.isfile('dllama')):
            os.system('make dllama')
        os.system(command)
