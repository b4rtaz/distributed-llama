# Distributed Llama

Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload.

## Measurements

### Raspberry Pi

#### Average Single Token Generation Time

All tests below utilized Q40 weights and a Q80 buffer. The generation time encompasses the inference time, network transfer time, sampling time, and multi-thread synchronization time. Number of samples: 64. All Raspberry Pi units were connected via Gigabit Ethernet to the TP-Link LS1008G Switch.

| Model       | 1 x RasPi 4B 8 GB                                                   | 2 x RasPi 4B 8 GB                                                     | 4 x RasPi 4B 8 GB                                                                    |
|-------------|---------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Llama 2 7B  | **1322.28 ms**<br><sub><sup>(I: 1248.00 ms, T: 6.17 ms)</sup></sub> | **825.00 ms**<br><sub><sup>(I: 658.84 ms, T: 98.45 ms)</sup></sub>    | **511.69 ms** `🔥 258% faster` <br><sub><sup>(I: 377.06 ms, T: 67.39 ms)</sup></sub> |
| Llama 2 13B | Not enough RAM                                                      | **1443.23 ms**<br><sub><sup>(I: 1299.30 ms, T: 58.59 ms)</sup></sub>  | **836.69 ms** <br><sub><sup>(I: 708.69 ms, T: 43.27 ms)</sup></sub>                  |

I - inference time of the root node.<br>
T - network transfer time.

#### Network Transfer for Generating Single Token

All tests below were conducted on 2 x Raspberry Pi 4B 8 GB units.

| Model       | F32 Buffer                                                       | Q80 Buffer                                                        |
|-------------|------------------------------------------------------------------|-------------------------------------------------------------------|
| Llama 2 7B  | **4880 kB**<br><sub><sup>(S: 2912 kB, R: 1968 kB)</sup></sub>    | **1295 kB** <br><sub><sup>(S: 773 kB, R: 522 kB)</sup></sub>      |
| Llama 2 13B | **7640 kB**<br><sub><sup>(S: 4560 kB, R: 3080 kB)</sup></sub>    | **2029 kB** <br><sub><sup>(S: 1211 kB, R: 818 kB)</sup></sub>     |

S - sent from the root node.<br>
R - received by the root node.

### MacBook Pro M1

70B:

```
⏩ Loaded 39706066944 bytes
🔶 G 293682 ms I 293091 ms T  137 ms S      0 kB R      0 kB Hello
🔶 G 400083 ms I 398644 ms T  171 ms S      0 kB R      0 kB  world
🔶 G 513173 ms I 511657 ms T  264 ms S      0 kB R      0 kB !
```

## How to Convert Llama 2 Weights

1. Download [Llama 2](https://github.com/facebookresearch/llama) weights. This project supports 7B, 13B and 70B models. This project doesn't support chat models.
2. Open the `llama-2-7b/params.json` file and replace `"vocab_size": -1` to `"vocab_size": 32000`.
3. Install dependencies of the converter `cd converter && pip install -r requirements.txt`.
4. Convert weights to Distributed Llama format: `python converter.py /path/to/llama-2-7b q40`. This will take a bit of time.
5. Compile Distributed Llama: `make main`.
6. Run Distributed Llama: `./main inference --model ./converter/dllama_llama-2-7b_q40.bin --tokenizer ../tokenizer.bin --weights-float-type q40 --buffer-float-type q80 --prompt "Hello world" --nthreads 4`.

In the table below, you can find the expected size of the converted weights with different floating-point types.

| Model       | Original size | Float32  | Float16  | Q40      |
|-------------|---------------|----------|----------|----------|
| Llama 2 7B  | 13.48 GB      |          |          | 3.95 GB  |
| Llama 2 13B | 26.03 GB      |          |          | 7.35 GB  |
| Llama 2 70B | 137.97 GB     |          |          | 36.98 GB |
