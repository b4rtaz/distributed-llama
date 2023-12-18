rm *.o
rm matmul-test
rm transformer-block-test
rm main

make quants
make matmul
make funcs
make matmul-test
chmod +x matmul-test
./matmul-test

make shared-buffer
make transformer
make transformer-block-test
chmod +x transformer-block-test
./transformer-block-test

make tokenizer
make worker
make main
chmod +x main
./main inference -m ./converter/llama_7b_q40.bin -f 2 -t /Users/b4rtaz/Dev/llama2.c/tokenizer.bin -prompt "Hello world" -nthread 4
#./main worker -p 9999
