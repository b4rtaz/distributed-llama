rm matmul.o
rm matmul-test
rm funcs.o
rm shared-buffer.o
rm transformer-block.o
rm transformer-block-test

make matmul
make matmul-test
make funcs
make shared-buffer
make transformer-block
make transformer-block-test
chmod +x matmul-test
chmod +x transformer-block-test
./transformer-block-test
