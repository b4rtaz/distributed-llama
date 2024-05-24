CXX = g++
CXXFLAGS = -std=c++11 -Werror -O3 -march=native -mtune=native

# Default settings
INCLUDES =
LIBPATH =
LIBS = -lpthread
EXTRA_COMMANDS = 

# Conditional settings for Windows
ifdef WIN32
    INCLUDES = -Isrc/common/pthreads-w32/include
    LIBPATH = -Lsrc/common/pthreads-w32/lib/x64
    LIBS = -lpthreadVC2 -lws2_32 # or -lpthreadGC2 if needed
    EXTRA_COMMANDS += copy src\common\pthreads-w32\dll\x64\pthreadVC2.dll $(BIN_DIR)\pthreadVC2.dll
endif

utils: src/utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/utils.cpp -o utils.o
quants: src/quants.cpp
	$(CXX) $(CXXFLAGS) -c src/quants.cpp -o quants.o
funcs: src/funcs.cpp
	$(CXX) $(CXXFLAGS) -c src/funcs.cpp -o funcs.o
funcs-test: src/funcs-test.cpp funcs
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o funcs-test funcs.o
socket: src/socket.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/socket.cpp -o socket.o
transformer: src/utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/transformer.cpp -o transformer.o
tasks: src/tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/tasks.cpp -o tasks.o
llama2-tasks: src/llama2-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/llama2-tasks.cpp -o llama2-tasks.o
grok1-tasks: src/grok1-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/grok1-tasks.cpp -o grok1-tasks.o
mixtral-tasks: src/mixtral-tasks.cpp
	$(CXX) $(CXXFLAGS) -c src/mixtral-tasks.cpp -o mixtral-tasks.o
tokenizer: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/tokenizer.cpp -o tokenizer.o
app: src/app.cpp
	$(CXX) $(CXXFLAGS) -c src/app.cpp -o app.o

dllama: src/apps/dllama/dllama.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer app
	$(CXX) $(CXXFLAGS) src/apps/dllama/dllama.cpp -o bin/dllama utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o app.o $(LIBPATH) $(LIBS)
	$(EXTRA_COMMANDS)

dllama-api: src/apps/dllama-api/dllama-api.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks mixtral-tasks tokenizer app
	$(CXX) $(CXXFLAGS) src/apps/dllama-api/dllama-api.cpp -o bin/dllama-api utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o mixtral-tasks.o tokenizer.o app.o $(LIBPATH) $(LIBS)
	$(EXTRA_COMMANDS)

funcs-test: src/funcs-test.cpp funcs utils quants
	$(CXX) $(CXXFLAGS) src/funcs-test.cpp -o bin/funcs-test funcs.o utils.o quants.o $(LIBPATH) $(LIBS)
quants-test: src/quants.cpp utils quants
	$(CXX) $(CXXFLAGS) src/quants-test.cpp -o bin/quants-test utils.o quants.o $(LIBPATH) $(LIBS)
transformer-test: src/transformer-test.cpp funcs utils quants transformer socket
	$(CXX) $(CXXFLAGS) src/transformer-test.cpp -o bin/transformer-test funcs.o utils.o quants.o transformer.o socket.o $(LIBPATH) $(LIBS)
llama2-tasks-test: src/llama2-tasks-test.cpp utils quants funcs socket transformer tasks llama2-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/llama2-tasks-test.cpp -o bin/llama2-tasks-test utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o tokenizer.o $(LIBPATH) $(LIBS)
grok1-tasks-test: src/grok1-tasks-test.cpp utils quants funcs socket transformer tasks llama2-tasks grok1-tasks tokenizer
	$(CXX) $(CXXFLAGS) src/grok1-tasks-test.cpp -o bin/grok1-tasks-test utils.o quants.o funcs.o socket.o transformer.o tasks.o llama2-tasks.o grok1-tasks.o tokenizer.o $(LIBPATH) $(LIBS)

clean:
	rm -f $(BIN_DIR)/*