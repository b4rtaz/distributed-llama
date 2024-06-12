CXX = g++
CXXFLAGS = -std=c++11 -Werror -O3 -march=native -mtune=native

# Conditional settings for Windows
ifeq ($(OS),Windows_NT)
    LIBS = -lws2_32 # or -lpthreadGC2 if needed
else
    LIBS = -lpthread
endif

utils: src/utils.cpp
	$(CXX) $(CXXFLAGS) -c src/utils.cpp -o utils.o
quants: src/quants.cpp
	$(CXX) $(CXXFLAGS) -c src/quants.cpp -o quants.o
funcs: src/funcs.cpp
	$(CXX) $(CXXFLAGS) -c src/funcs.cpp -o funcs.o
funcs-test: src/funcs-test.cpp funcs
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o funcs-test funcs.o
commands: src/commands.cpp
	$(CXX) $(CXXFLAGS) -c src/commands.cpp -o commands.o
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
app: src/app.cpp
	$(CXX) $(CXXFLAGS) -c src/app.cpp -o app.o

dllama: src/apps/dllama/dllama.cpp utils quants funcs commands socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer app
	$(CXX) $(CXXFLAGS) src/apps/dllama/dllama.cpp -o dllama utils.o quants.o funcs.o commands.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o app.o $(LIBS)
dllama-api: src/apps/dllama-api/dllama-api.cpp utils quants funcs commands socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer app
	$(CXX) $(CXXFLAGS) src/apps/dllama-api/dllama-api.cpp -o dllama-api utils.o quants.o funcs.o commands.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o app.o $(LIBS)

funcs-test: src/funcs-test.cpp funcs utils quants
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o funcs-test funcs.o utils.o quants.o $(LIBS)
quants-test: src/quants.cpp utils quants
	$(CXX) $(CXXFLAGS) src/quants-test.cpp -o quants-test utils.o quants.o $(LIBS)
tokenizer-test: src/tokenizer-test.cpp tokenizer funcs commands utils quants
	$(CXX) $(CXXFLAGS) src/tokenizer-test.cpp -o tokenizer-test tokenizer.o funcs.o commands.o utils.o quants.o $(LIBS)
commands-test: src/commands-test.cpp funcs commands utils quants transformer socket
	$(CXX) $(CXXFLAGS) src/commands-test.cpp -o commands-test funcs.o commands.o utils.o quants.o transformer.o socket.o $(LIBS)
llama2-tasks-test: src/llama2-tasks-test.cpp utils quants funcs commands socket transformer tasks llama2-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/llama2-tasks-test.cpp -o llama2-tasks-test utils.o quants.o funcs.o commands.o socket.o transformer.o tasks.o llama2-tasks.o tokenizer.o $(LIBS)
grok1-tasks-test: src/grok1-tasks-test.cpp utils quants funcs commands socket transformer tasks llama2-tasks grok1-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/grok1-tasks-test.cpp -o grok1-tasks-test utils.o quants.o funcs.o commands.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o tokenizer.o $(LIBS)