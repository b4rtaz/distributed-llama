# Distributed Llama API

This is an early version of the server that is compatible with the OpenAi API. It supports only the `/v1/chat/completions` endpoint. To run this server you need a chat model and a tokenizer with the chat support.

How to run?

1. Download the model and the tokenizer from [here](https://huggingface.co/b4rtaz/Llama-3-8B-Q40-Instruct-Distributed-Llama).
2. Run the server with the following command:
```bash
./dllama-api --model converter/dllama_model_lama3_instruct_q40.m --tokenizer converter/dllama_tokenizer_llama3.t --weights-float-type q40 --buffer-float-type q80 --nthreads 4
```

Check the [chat-api-client.js](../../../examples/chat-api-client.js) file to see how to use the API from NodeJS application.
