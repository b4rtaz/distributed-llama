make quants-test
chmod +x ./quants-test
./quants-test

make transformer-tasks-test
chmod +x ./transformer-tasks-test
./transformer-tasks-test

make main
chmod +x ./main
./main inference --model ./converter/dllama_llama-2-7b_q40.bin --weights-float-type q40 --buffer-float-type q80 --tokenizer /Users/b4rtaz/Dev/llama2.c/tokenizer.bin --prompt "Hello world" --nthreads 4
