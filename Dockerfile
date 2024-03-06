# https://vsupalov.com/docker-arg-env-variable-guide/
# docker build arg to all FROM

ARG A_ALPINE_IMG_TAG=latest
FROM alpine:${A_ALPINE_IMG_TAG} as base

# docker img meta
LABEL dllama.image.authors="weege007@gmail.com"

# docker build arg after each FROM
ARG A_WORKER_PARAM="--port 9997 --nthreads 1"
ARG A_INFERENCE_PARAM="--prompt 'Hello' --steps 1 --nthreads 1 --workers 127.0.0.1:9997"
ARG A_GIT_REPO_URL=https://github.com/b4rtaz/distributed-llama.git
# container env
ENV E_WORKER_PARAM ${A_WORKER_PARAM}
ENV E_INFERENCE_PARAM ${A_INFERENCE_PARAM}

# build prepare layer, use arg/env
RUN set -eux; \
    \
    apk add --no-cache \
    git \
    g++ \
    make \
    python3 \
    py3-pip \
    ; \
    git clone ${A_GIT_REPO_URL} distributed-llama; \
    make -C distributed-llama main; \
    \
    echo "Compile Distributed Llama Done\n"

# Custom cache invalidation
ARG CACHEBUST=1

FROM base as download_bin
ARG A_HF_ENDPOINT="https://huggingface.co"
ARG A_CHECKPOINT_URL=${A_HF_ENDPOINT}/karpathy/tinyllamas/resolve/main/stories42M.bin
ARG A_TOKEN_URL=https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
# convert checkpoint/tokenizer, use arg
RUN cd distributed-llama;\
    wget -O tokenizer.bin "${A_TOKEN_URL}"; \
    wget -O model.bin "${A_CHECKPOINT_URL}"; \
    python3 converter/convert-legacy.py model.bin true; \
    ls -agl ./; \
    \
    echo "convert-legacy Done\n"

# need local download bin file to container
FROM base as copy_bin
# convert checkpoint/tokenizer 
# from local download bin file to container
COPY tokenizer.bin /distributed-llama
COPY model.bin /distributed-llama
RUN cd distributed-llama;\
    python3 converter/convert-legacy.py model.bin true; \
    ls -agl ./; \
    \
    echo "convert-legacy Done\n"

# docker run container runtime, use env
FROM base as worker
RUN echo "E_WORKER_PARAM: ${E_WORKER_PARAM}\n"
CMD ["sh","-c","/distributed-llama/main worker ${E_WORKER_PARAM}"]

FROM download_bin as inference
RUN echo "E_INFERENCE_PARAM: ${E_INFERENCE_PARAM}\n"
CMD ["sh","-c","/distributed-llama/main inference --model /distributed-llama/dllama_model.bin --tokenizer /distributed-llama/tokenizer.bin ${E_INFERENCE_PARAM}"]

# need local download bin file to container
FROM copy_bin as inference_local
RUN echo "E_INFERENCE_PARAM: ${E_INFERENCE_PARAM}\n"
CMD ["sh","-c","/distributed-llama/main inference --model /distributed-llama/dllama_model.bin --tokenizer /distributed-llama/tokenizer.bin ${E_INFERENCE_PARAM}"]
