rm matmul
rm matmul-test
rm funcs
rm transformer
rm transformer-test

make matmul
make matmul-test
make funcs
make transformer
make transformer-test
chmod +x matmul-test
chmod +x transformer-test
./transformer-test
