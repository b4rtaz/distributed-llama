CXX = g++
CXXFLAGS = -std=c++11 -Werror -O3 -march=native -mtune=native

utils: src/utils.cpp
	$(CXX) $(CXXFLAGS) -c src/utils.cpp -o utils.o
quants: src/quants.cpp
	$(CXX) $(CXXFLAGS) -c src/quants.cpp -o quants.o
funcs: src/funcs.cpp
	$(CXX) $(CXXFLAGS) -c src/funcs.cpp -o funcs.o
funcs-test: src/funcs-test.cpp funcs
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o funcs-test funcs.o
socket: src/socket.cpp
	$(CXX) $(CXXFLAGS) -c src/socket.cpp -o socket.o
transformer: src/utils.cpp
	$(CXX) $(CXXFLAGS) -c src/transformer.cpp -o transformer.o
tasks: src/tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/tasks.cpp -o tasks.o
llama2-tasks: src/llama2-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/llama2-tasks.cpp -o llama2-tasks.o
grok1-tasks: src/grok1-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/grok1-tasks.cpp -o grok1-tasks.o
mixtral-tasks: src/mixtral-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/mixtral-tasks.cpp -o mixtral-tasks.o
tokenizer: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c src/tokenizer.cpp -o tokenizer.o
http: src/http.cpp
	$(CXX) $(CXXFLAGS) -c src/http.cpp -o http.o

main: src/main.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/main.cpp -o main utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o -lpthread

server: src/server.cpp http utils quants funcs socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/server.cpp -o server http.o utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o -lpthread
funcs-test: src/funcs-test.cpp funcs utils quants
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o funcs-test funcs.o utils.o quants.o -lpthread
quants-test: src/quants.cpp utils quants
	$(CXX) $(CXXFLAGS) src/quants-test.cpp -o quants-test utils.o quants.o -lpthread
transformer-test: src/transformer-test.cpp funcs utils quants transformer socket
	$(CXX) $(CXXFLAGS) src/transformer-test.cpp -o transformer-test funcs.o utils.o quants.o transformer.o socket.o -lpthread
llama2-tasks-test: src/llama2-tasks-test.cpp utils quants funcs socket transformer tasks llama2-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/llama2-tasks-test.cpp -o llama2-tasks-test utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o tokenizer.o -lpthread
grok1-tasks-test: src/grok1-tasks-test.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/grok1-tasks-test.cpp -o grok1-tasks-test utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o tokenizer.o -lpthread
