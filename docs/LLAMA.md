# How to Run Llama

## How to Run Llama 2

1. Download [Llama 2](https://github.com/facebookresearch/llama) weights from Meta. This project supports 7B, 7B-chat, 13B, 13B-chat, 70B and 70B-chat models.
2. Open the `llama-2-7b/params.json` file:
  * replace `"vocab_size": -1` to `"vocab_size": 32000`,
  * add a new property: `"max_seq_len": 2048`.
3. Install dependencies of the converter:
```sh
cd converter && pip install -r requirements.txt
```
4. Convert weights to Distributed Llama format. This will take a bit of time. The script requires Python 3.
```sh
python convert-llama.py /path/to/meta/llama-2-7b q40
```
5. Download the tokenizer for Llama 2:
```
wget https://huggingface.co/b4rtaz/Llama-2-Tokenizer-Distributed-Llama/resolve/main/dllama_tokenizer_llama2.t
```
6. Build the project:
```bash
make dllama
make dllama-api
```
7. Run:
```bash
./dllama inference --model dllama_llama-2-7b_q40.bin --tokenizer dllama-llama2-tokenizer.t --weights-float-type q40 --buffer-float-type q80 --prompt "Hello world" --steps 16 --nthreads 4
```

In the table below, you can find the expected size of the converted weights with different floating-point types.

| Model       | Original size | Float32  | Float16  | Q40      |
|-------------|---------------|----------|----------|----------|
| Llama 2 7B  | 13.48 GB      | 25.10GB  |          | 3.95 GB  |
| Llama 2 13B | 26.03 GB      |          |          | 7.35 GB  |
| Llama 2 70B | 137.97 GB     |          |          | 36.98 GB |

## How to Run Llama 3

1. Get an access to the model on [Llama 3 website](https://llama.meta.com/llama-downloads).
2. Clone the `https://github.com/meta-llama/llama3` repository.
3. Run the `download.sh` script to download the model.
4. For Llama 3 8B model you should have the following files:
    - `Meta-Llama-3-8B/consolidated.00.pth`
    - `Meta-Llama-3-8B/params.json`
    - `Meta-Llama-3-8B/tokenizer.model`
5. Open `params.json` and add a new property: `"max_seq_len": 8192`.
6. Clone the `https://github.com/b4rtaz/distributed-llama.git` repository.
7. Install dependencies of the converter:
```sh
cd converter && pip install -r requirements.txt
```
8. Convert the model to the Distributed Llama format:
```bash
python converter/convert-llama.py path/to/Meta-Llama-3-8B q40
```
9. Convert the tokenizer to the Distributed Llama format:
```bash
python converter/convert-tokenizer-llama3.py path/to/tokenizer.model
```
10. Build the project:
```bash
make dllama
make dllama-api
```
11. Run the Distributed Llama:
```bash
./dllama inference --weights-float-type q40 --buffer-float-type q80 --prompt "My name is" --steps 128 --nthreads 8 --model dllama_meta-llama-3-8b_q40.bin --tokenizer llama3-tokenizer.t
```
