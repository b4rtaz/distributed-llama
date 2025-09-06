# How to Convert 🤗 Hugging Face Model

Currently, Distributed Llama supports these Hugging Face models: `llama`, `mistral`, `qwen3` and `qwen3_moe`. You can try to convert any compatible Hugging Face model and run it with Distributed Llama.

> [!IMPORTANT]
> All converters are in the early stages of development. After conversion, the model may not work correctly.

1. Download a model, for example: [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/tree/main).
2. The downloaded model should contain `config.json`, `tokenizer.json`, `tokenizer_config.json` and `tokenizer.model` and safetensor files.
3. Run the converter of the model:

```sh
cd converter
python convert-hf.py path/to/hf/model q40 mistral-7b-0.3
```

4. Run the converter of the tokenizer:

```sh
python convert-tokenizer-hf.py path/to/hf/model mistral-7b-0.3
```

5. That's it! Now you can run the Distributed Llama.

```sh
./dllama inference \
  --prompt "Hello world" \
  --steps 64 \
  --model dllama_model_mistral-7b-0.3_q40.m \
  --tokenizer dllama_tokenizer_mistral-7b-0.3.t \
  --buffer-float-type q80
```
