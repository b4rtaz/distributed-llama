rm matmul.o
rm -rf matmul-test
make matmul
make matmul-test
chmod +x matmul-test
./matmul-test
