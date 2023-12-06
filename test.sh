rm matmul
rm matmul-test
rm funcs
rm shared-buffer
rm transformer
rm transformer-test

make matmul
make matmul-test
make funcs
make shared-buffer
make transformer
make transformer-test
chmod +x matmul-test
chmod +x transformer-test
./transformer-test
