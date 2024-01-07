# Distributed Llama



## How to Convert Llama 2 Weights

1. Download [Llama 2](https://github.com/facebookresearch/llama) weights. This project supports 7B, 13B and 70B models. This project doesn't support chat models.
2. Open the `llama-2-7b/params.json` file and replace `"vocab_size": -1` to `"vocab_size": 32000`.
3. Install dependencies of the converter `cd converter && pip install -r requirements.txt`.
4. Convert weights to Distributed Llama format: `python converter.py /path/to/llama-2-7b q40`. This will take a bit of time.
5. Compile Distributed Llama: `make main`.
6. Run Distributed Llama: `./main inference -m ./converter/dllama_llama-2-7b_q40.bin -w 2 -b 3 -t ../tokenizer.bin -prompt "Hello world" -nthread 4`.
