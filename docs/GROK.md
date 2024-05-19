# How to Run Grok-1

1. Download quantized (Q40) weights from https://huggingface.co/b4rtaz/grok-1-dllama (180GB).
2. Merge split models files into single file:
```
cat dllama-grok-1-q40.binaa dllama-grok-1-q40.binab dllama-grok-1-q40.binac dllama-grok-1-q40.binad dllama-grok-1-q40.binae dllama-grok-1-q40.binaf dllama-grok-1-q40.binag dllama-grok-1-q40.binah dllama-grok-1-q40.binai > dllama-grok-1-q40-final.bin
```
3. Download the tokenizer:
```
wget https://huggingface.co/b4rtaz/grok-1-distributed-llama/resolve/main/dllama-grok1-tokenizer.t
```
4. Build the project:
```bash
make dllama
```
5. Run the model:
```bash
./dllama inference --weights-float-type q40 --buffer-float-type q80 --prompt "Hello" --steps 128 --nthreads 8 --model dllama-grok-1-q40-final.bin --tokenizer dllama-grok1-tokenizer.t
```