CXX = g++
CXXFLAGS = -std=c++11 -Werror -Wformat -Werror=format-security
ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
endif

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
nn-quants.o: src/nn/nn-quants.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-core.o: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-executor.o: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-network.o: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llamafile-sgemm.o: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu-ops.o: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu.o: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

# llm
tokenizer.o: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llm.o: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
app.o: src/app.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
dllama: src/dllama.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
dllama-api: src/dllama-api.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
