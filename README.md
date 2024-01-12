# Distributed Llama

Run LLMs on weak devices or make powerful devices even more powerful by distributing the workload.

## 📊 Measurements

### Raspberry Pi

#### Average Single Token Generation Time

All tests below utilized Q40 weights and a Q80 buffer. The generation time encompasses the inference time, network transfer time, sampling time, and multi-thread synchronization time. Number of samples: 16. All Raspberry Pi units were connected via Gigabit Ethernet to the TP-Link LS1008G Switch.

| Model       | 1 x RasPi 4B 8 GB                                                   | 2 x RasPi 4B 8 GB                                                     | 4 x RasPi 4B 8 GB                                                                    |
|-------------|---------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Llama 2 7B  | **1312.50 ms**<br><sub><sup>(I: 1307.94 ms, T: 1.81 ms)</sup></sub> | **793.69 ms**<br><sub><sup>(I: 739.00 ms, T: 52.50 ms)</sup></sub>    | **494.00 ms** `🔥 265% faster` <br><sub><sup>(I: 458.81 ms, T: 34.06 ms)</sup></sub> |
| Llama 2 13B | Not enough RAM                                                      | **1497.19 ms**<br><sub><sup>(I: 1465.06 ms, T: 30.88 ms)</sup></sub>  | **848.19 ms** <br><sub><sup>(I: 746.88 ms, T: 99.50 ms)</sup></sub>                  |

I - inference time of the root node.<br>
T - network transfer time.

#### Network Transfer for Generating Single Token

All tests below were conducted on 2 x Raspberry Pi 4B 8 GB units.

| Model       | F32 Buffer                                                       | Q80 Buffer                                                        |
|-------------|------------------------------------------------------------------|-------------------------------------------------------------------|
| Llama 2 7B  | **4192 kB**<br><sub><sup>(S: 2224 kB, R: 1968 kB)</sup></sub>    | **1112 kB** <br><sub><sup>(S: 590 kB, R: 522 kB)</sup></sub>      |
| Llama 2 13B | **6560 kB**<br><sub><sup>(S: 3480 kB, R: 3080 kB)</sup></sub>    | **1742 kB** <br><sub><sup>(S: 924 kB, R: 818 kB)</sup></sub>     |

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

## 🔨 How to Convert Llama 2 Weights

1. Download [Llama 2](https://github.com/facebookresearch/llama) weights. This project supports 7B, 13B and 70B models. This project doesn't support chat models.
2. Open the `llama-2-7b/params.json` file and replace `"vocab_size": -1` to `"vocab_size": 32000`.
3. Install dependencies of the converter:
```sh
cd converter && pip install -r requirements.txt`
```
5. Convert weights to Distributed Llama format. This will take a bit of time.
```sh
python converter.py /path/to/llama-2-7b q40
```
7. Compile Distributed Llama:
```sh
make main
```
9. Run Distributed Llama:
```sh
./main inference --model ../dllama_llama-2-13b_q40.bin --tokenizer ../tokenizer.bin --weights-float-type q40 --buffer-float-type q80 --prompt "Hello world" --steps 16 --nthreads 4
```

In the table below, you can find the expected size of the converted weights with different floating-point types.

| Model       | Original size | Float32  | Float16  | Q40      |
|-------------|---------------|----------|----------|----------|
| Llama 2 7B  | 13.48 GB      |          |          | 3.95 GB  |
| Llama 2 13B | 26.03 GB      |          |          | 7.35 GB  |
| Llama 2 70B | 137.97 GB     |          |          | 36.98 GB |

## 📟 How to Run on Raspberry Pi Devices

1. Install `Raspberry Pi OS Lite (64 bit)` on your Raspberry Pi devices. This OS doesn't have desktop environment.
2. Connect to all devices via SSH.
```
ssh user@raspberrypi1.local
ssh user@raspberrypi2.local
```
3. Install Git:
```sh
sudo apt install git
```
4. Clone this repository:
```sh
git clone https://github.com/b4rtaz/distributed-llama.git
```
5. Compile Distributed Llama:
```sh
make main
```
6. Optional: assign static IP addresses.
```sh
sudo ip addr add 10.0.0.1/24 dev eth0 # 1th device
sudo ip addr add 10.0.0.2/24 dev eth0 # 2th device
```
7. Run worker nodes on worker devices:
```
sudo nice -n -20 ./main worker --port 9998
```
8. Run root node on main device:
```sh
sudo nice -n -20 ./main inference --model ../dllama_llama-2-13b_q40.bin --tokenizer ../tokenizer.bin --weights-float-type q40 --buffer-float-type q80 --prompt "Hello world" --steps 16 --nthreads 4 --workers 10.0.0.1:9998
```

## 💡 License

This project is released under the MIT license.