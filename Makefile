CC = gcc

matmul-test: matmul/matmul-test.c
	$(CC) -O3 matmul/matmul-test.c -g -o matmul-test
