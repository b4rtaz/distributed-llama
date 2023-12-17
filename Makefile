CXX = g++
CXXFLAGS = -std=c++11 -Werror -O3

quants: src/quants.cpp
	$(CXX) $(CXXFLAGS) -c -pthread src/quants.cpp -o quants.o

matmul: src/matmul.cpp quants
	$(CXX) $(CXXFLAGS) -c -pthread src/matmul.cpp -o matmul.o

funcs: src/funcs.cpp
	$(CXX) $(CXXFLAGS) -c src/funcs.cpp -o funcs.o

matmul-test: src/matmul-test.cpp
	$(CXX) $(CXXFLAGS) src/matmul-test.cpp -o matmul-test quants.o matmul.o funcs.o

shared-buffer: src/shared-buffer.cpp
	$(CXX) $(CXXFLAGS) -c src/shared-buffer.cpp -o shared-buffer.o

transformer: src/transformer.cpp shared-buffer matmul
	$(CXX) $(CXXFLAGS) -c src/transformer.cpp -o transformer.o

transformer-block-test: src/transformer-block-test.cpp
	$(CXX) $(CXXFLAGS) src/transformer-block-test.cpp -o transformer-block-test shared-buffer.o transformer.o quants.o matmul.o funcs.o

tokenizer: src/tokenizer.cpp transformer
	$(CXX) $(CXXFLAGS) -c src/tokenizer.cpp -o tokenizer.o

worker: src/worker.cpp transformer
	$(CXX) $(CXXFLAGS) -c src/worker.cpp -o worker.o

main: src/main.cpp worker shared-buffer transformer quants matmul funcs tokenizer
	$(CXX) $(CXXFLAGS) src/main.cpp -o main shared-buffer.o worker.o transformer.o quants.o matmul.o funcs.o tokenizer.o
