CC = g++

matmul: src/matmul.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/matmul.cpp -o matmul.o

matmul-test: src/matmul-test.cpp matmul.o
	$(CC) -std=c++11 -Werror -O3 src/matmul-test.cpp -g -o matmul-test matmul.o

funcs: src/funcs.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/funcs.cpp -o funcs.o

shared-buffer: src/shared-buffer.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/shared-buffer.cpp -o shared-buffer.o

transformer-block: src/transformer-block.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/transformer-block.cpp -g -o transformer-block.o

transformer-block-test: src/transformer-block-test.cpp
	$(CC) -std=c++11 -Werror -O3 src/transformer-block-test.cpp -g -o transformer-block-test shared-buffer.o transformer-block.o matmul.o funcs.o
