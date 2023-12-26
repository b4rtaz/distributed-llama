make transformer-tasks-test
chmod +x ./transformer-tasks-test
./transformer-tasks-test

make main
chmod +x ./main
./main inference -m ./converter/llama_7b_q40.bin -f 2 -t /Users/b4rtaz/Dev/llama2.c/tokenizer.bin -prompt "Hello world" -nthread 4
