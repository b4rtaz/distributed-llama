CXX = g++
CXXFLAGS = -std=c++11 -Werror -Wformat -Werror=format-security

ifdef DEBUG
	CXXFLAGS += -g
else
	CXXFLAGS += -O3
endif
ifdef WVLA
	CXXFLAGS += -Wvla-extension
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

# nn
nn-quants:
	$(CXX) $(CXXFLAGS) -c src/nn/nn-quants.cpp -o nn-quants.o
nn-core: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-core.cpp -o nn-core.o
nn-executor: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-executor.cpp -o nn-executor.o
nn-network: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-network.cpp -o nn-network.o
llamafile-sgemm: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/llamafile/sgemm.cpp -o llamafile-sgemm.o
nn-cpu-ops: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-cpu-ops.cpp -o nn-cpu-ops.o
nn-cpu: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c src/nn/nn-cpu.cpp -o nn-cpu.o
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants nn-core nn-executor llamafile-sgemm nn-cpu-ops nn-cpu
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-test.cpp -o nn-cpu-test nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants nn-core llamafile-sgemm
	$(CXX) $(CXXFLAGS) src/nn/nn-cpu-ops-test.cpp -o nn-cpu-ops-test nn-quants.o nn-core.o llamafile-sgemm.o $(LIBS)

# llm
tokenizer: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c src/tokenizer.cpp -o tokenizer.o
llm: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c src/llm.cpp -o llm.o
app: src/app.cpp
	$(CXX) $(CXXFLAGS) -c src/app.cpp -o app.o
dllama: src/dllama.cpp nn-quants nn-core nn-executor nn-network llamafile-sgemm nn-cpu-ops nn-cpu tokenizer llm app
	$(CXX) $(CXXFLAGS) src/dllama.cpp -o dllama nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o $(LIBS)
dllama-api: src/dllama-api.cpp nn-quants nn-core nn-executor nn-network llamafile-sgemm nn-cpu-ops nn-cpu tokenizer llm app
	$(CXX) $(CXXFLAGS) src/dllama-api.cpp -o dllama-api nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o $(LIBS)
