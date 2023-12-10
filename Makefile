CC = g++

quants: src/quants.cpp
	$(CC) -std=c++11 -Werror -O3 -c -pthread src/quants.cpp -o quants.o

matmul: src/matmul.cpp
	$(CC) -std=c++11 -Werror -O3 -c -pthread src/matmul.cpp -o matmul.o

funcs: src/funcs.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/funcs.cpp -o funcs.o

matmul-test: src/matmul-test.cpp
	$(CC) -std=c++11 -Werror -O3 src/matmul-test.cpp -g -o matmul-test quants.o matmul.o funcs.o

shared-buffer: src/shared-buffer.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/shared-buffer.cpp -o shared-buffer.o

transformer: src/transformer.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/transformer.cpp -g -o transformer.o

transformer-block-test: src/transformer-block-test.cpp
	$(CC) -std=c++11 -Werror -O3 src/transformer-block-test.cpp -g -o transformer-block-test shared-buffer.o transformer.o quants.o matmul.o funcs.o

tokenizer: src/transformer.cpp
	$(CC) -std=c++11 -Werror -O3 -c src/tokenizer.cpp -g -o tokenizer.o

main: src/main.cpp
	$(CC) -std=c++11 -Werror -O3 src/main.cpp -g -o main shared-buffer.o transformer.o quants.o matmul.o funcs.o tokenizer.o
