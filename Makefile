CC = g++

matmul: src/matmul.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/matmul.cpp -o matmul.o

matmul-test: src/matmul-test.cpp matmul.o
	$(CC) -std=c++11 -Werror -O3 src/matmul-test.cpp -g -o matmul-test matmul.o
