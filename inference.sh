make quants-test
chmod +x ./quants-test
./quants-test

make transformer-tasks-test
chmod +x ./transformer-tasks-test
./transformer-tasks-test

make main
chmod +x ./main
./main inference -m ./converter/dllama_llama-2-7b_q40.bin -w 2 -b 3 -t /Users/b4rtaz/Dev/llama2.c/tokenizer.bin -prompt "Hello world" -nthread 4
