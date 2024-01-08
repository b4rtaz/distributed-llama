# Distributed Llama



## How to Convert Llama 2 Weights

1. Download [Llama 2](https://github.com/facebookresearch/llama) weights. This project supports 7B, 13B and 70B models. This project doesn't support chat models.
2. Open the `llama-2-7b/params.json` file and replace `"vocab_size": -1` to `"vocab_size": 32000`.
3. Install dependencies of the converter `cd converter && pip install -r requirements.txt`.
4. Convert weights to Distributed Llama format: `python converter.py /path/to/llama-2-7b q40`. This will take a bit of time.
5. Compile Distributed Llama: `make main`.
6. Run Distributed Llama: `./main inference --model ./converter/dllama_llama-2-7b_q40.bin --tokenizer ../tokenizer.bin --weights-float-type q40 --buffer-float-type q80 --prompt "Hello world" --nthreads 4`.

In the table below, you can find the expected size of the converted weights with different floating-point types.

| Model       | Original size | Float32  | Float16  | Q40      |
|-------------|---------------|----------|----------|----------|
| Llama 2 7B  | 13,48 GB      |          |          | 3,95 GB  |
| Llama 2 13B | 26,03 GB      |          |          | 7,35 GB  |
| Llama 2 70B | 137,97 GB     |          |          | 36,98 GB |
