CXX = g++
CXXFLAGS = -std=c++11 -Werror -O3 -march=native -mtune=native

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
