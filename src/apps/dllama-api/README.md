# Distributed Llama API

This is an early version of the server that is compatible with the OpenAi API. It supports only the `/v1/chat/completions` endpoint. Currently it's adjusted to the Llama 3 8B Instruct only.

How to run?

1. Download the model and the tokenizer from [here](https://huggingface.co/Azamorn/Meta-Llama-3-8B-Instruct-Distributed).
2. Run the server with the following command:
```bash
./dllama-api --model converter/dllama_original_q40.bin --tokenizer converter/dllama-llama3-tokenizer.t --weights-float-type q40 --buffer-float-type q80 --nthreads 4
```

Check the [chat-api-client.js](../../../examples/chat-api-client.js) file to see how to use the API from NodeJS application.
