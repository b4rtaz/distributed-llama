CXX = g++
CXXFLAGS = -std=c++11 -Werror -O3 -march=native -mtune=native -g

utils: src/utils.cpp
	$(CXX) $(CXXFLAGS) -c src/utils.cpp -o utils.o
quants: src/quants.cpp
	$(CXX) $(CXXFLAGS) -c src/quants.cpp -o quants.o
funcs: src/funcs.cpp
	$(CXX) $(CXXFLAGS) -c src/funcs.cpp -o funcs.o
socket: src/socket.cpp
	$(CXX) $(CXXFLAGS) -c src/socket.cpp -o socket.o
transformer: src/utils.cpp
	$(CXX) $(CXXFLAGS) -c src/transformer.cpp -o transformer.o
transformer-tasks: src/transformer-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/transformer-tasks.cpp -o transformer-tasks.o
tokenizer: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c src/tokenizer.cpp -o tokenizer.o

main: src/main.cpp utils quants funcs socket transformer transformer-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/main.cpp -o main utils.o quants.o funcs.o socket.o transformer.o transformer-tasks.o tokenizer.o -lpthread
quants-test: src/quants.cpp utils quants
	$(CXX) $(CXXFLAGS) src/quants-test.cpp -o quants-test utils.o quants.o -lpthread
transformer-tasks-test: src/transformer-tasks-test.cpp utils quants funcs socket transformer transformer-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/transformer-tasks-test.cpp -o transformer-tasks-test utils.o quants.o funcs.o socket.o transformer.o transformer-tasks.o tokenizer.o -lpthread


docker-worker-build:
	@docker build -f Dockerfile -t alpine_dllama_worker \
		--target worker \
		--build-arg A_ALPINE_IMG_TAG=latest \
 		. 

docker-inference-build:
	@docker build -f Dockerfile -t alpine_dllama_inference \
		--target inference \
		--build-arg A_ALPINE_IMG_TAG=latest \
		.

docker-inference-build-hf-mirror:
	@docker build -f Dockerfile -t alpine_dllama_inference \
		--target inference \
		--build-arg A_ALPINE_IMG_TAG=latest \
		--build-arg A_HF_ENDPOINT="https://hf-mirror.com" \
 		. 

download_stories42M_bin:
	@wget -O tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
	@wget -O model.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin

# need local download tokenizer.bin model.bin
# like: download_stories42M_bin
docker-inference-build-local:
	@docker build -f Dockerfile -t alpine_dllama_inference \
		--target inference_local \
		--build-arg A_ALPINE_IMG_TAG=latest \
		.

docker_create_network:
	@docker network create -d bridge dllama-net

# if local host to run inference, add -p 9997:****
WORKER_ID = 1
docker-worker-run:
	@docker run -itd --rm \
		--name alpine-dllama-worker-$(WORKER_ID) \
		--network dllama-net \
		-e E_WORKER_PARAM="--port 9997 --nthreads 1" \
		alpine_dllama_worker

# workers list need a discover sys like etcd
WORKERS = 172.18.0.2:9997
docker-inference-run:
	@docker run -it --rm \
		--network dllama-net \
		--name alpine-dllama-inference \
		-e E_INFERENCE_PARAM="--prompt Hello --steps 16 --nthreads 1 --workers $(WORKERS)"\
		alpine_dllama_inference

docker-1-worker-inference:
	@make docker-worker-run 
	@make docker-inference-run

docker-3-worker-inference:
	@make docker-worker-run 
	@make docker-worker-run WORKER_ID=2
	@make docker-worker-run WORKER_ID=3
	@make docker-inference-run WORKERS="172.18.0.2:9997 172.18.0.3:9997 172.18.0.4:9997"