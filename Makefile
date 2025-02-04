CXX = g++
CXXFLAGS = -std=c++11 -Werror -march=native -mtune=native -Wformat -Wvla-extension -Werror=format-security

ifeq ($(shell uname -m),aarch64)
	CXXFLAGS += -mfp16-format=ieee
endif

ifdef DEBUG
	CXXFLAGS += -g
else
	CXXFLAGS += -O3
endif

ifeq ($(OS),Windows_NT)
    LIBS = -lws2_32
	DELETE_CMD = del /f
else
    LIBS = -lpthread
    DELETE_CMD = rm -fv
endif

.PHONY: clean dllama

clean:
	$(DELETE_CMD) *.o dllama dllama-* socket-benchmark mmap-buffer-* *-test *.exe
nn-core: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-core.cpp -o nn-core.o
nn-executor: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-executor.cpp -o nn-executor.o
nn-network: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-network.cpp -o nn-network.o
nn-cpu-ops: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-cpu-ops.cpp -o nn-cpu-ops.o
nn-cpu: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-cpu.cpp -o nn-cpu.o
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-core nn-executor nn-cpu-ops nn-cpu
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-test.cpp -o nn-cpu-test nn-core.o nn-executor.o nn-cpu-ops.o nn-cpu.o $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-core
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-ops-test.cpp -o nn-cpu-ops-test nn-core.o $(LIBS)

tokenizer: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c src/tokenizer.cpp -o tokenizer.o
llm: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c src/llm.cpp -o llm.o
app: src/app.cpp
	$(CXX) $(CXXFLAGS) -c src/app.cpp -o app.o

dllama: src/dllama.cpp nn-core nn-executor nn-network nn-cpu-ops nn-cpu tokenizer llm app
	$(CXX) $(CXXFLAGS) src/dllama.cpp -o dllama nn-core.o nn-executor.o nn-network.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o $(LIBS)
