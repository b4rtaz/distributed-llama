FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app
RUN make dllama && make dllama-api

# Default ports for root node + worker node
EXPOSE 5000
EXPOSE 9999

# TODO: Consider putting the binary on a smaller image layer
CMD ["./dllama"]
