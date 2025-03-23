![Distributed Llama](.github/cover.png)

# Distributed Llama

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/b4rtaz/distributed-llama/.github%2Fworkflows%2Fmain.yml?style=flat-square)](https://github.com/b4rtaz/distributed-llama/actions) [![License: MIT](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](/LICENSE) [![Support this project](https://img.shields.io/github/sponsors/b4rtaz?style=flat-square&label=support%20this%20project&color=green)](https://github.com/sponsors/b4rtaz) [![Discord](https://discordapp.com/api/guilds/1245814812353495070/widget.png?style=shield)](https://discord.com/widget?id=1245814812353495070&theme=dark)

Connect home devices into a powerful cluster to accelerate LLM inference. More devices mean faster performance, leveraging tensor parallelism and high-speed synchronization over Ethernet.

Supports Linux, macOS, and Windows. Optimized for ARM and x86_64 AVX2 CPUs.

**News**
- 23 Mar 2025 - [🌋 Experimental Vulkan support](https://github.com/b4rtaz/distributed-llama/releases/tag/v0.13.0)
- 12 Feb 2025 - 🚧 Merged the [fundamental codebase refactor](https://github.com/b4rtaz/distributed-llama/releases/tag/v0.12.0)
- 9 Jan 2025 - [🍎 Llama 3.3 70B on 4 x Mac Mini M4 Pro 24GB RAM](https://github.com/b4rtaz/distributed-llama/discussions/147)
- 28 Jul 2024 - [🌳 How to Run Llama 3.1 405B on Home Devices? Build AI Cluster!](https://medium.com/@b4rtaz/how-to-run-llama-3-405b-on-home-devices-build-ai-cluster-ad0d5ad3473b)


### 🔥 Setup Root Node by Single Command

Python 3 and C++ compiler required. The command will download the model and the tokenizer.

| Model                             | Size     | Command                                              |
| --------------------------------- | -------- | ---------------------------------------------------- |
| Llama 3.1 8B Instruct Q40         | 6.32 GB  | `python launch.py llama3_1_8b_instruct_q40`          |
| Llama 3.1 405B Instruct Q40.      | 238 GB   | `python launch.py llama3_1_405b_instruct_q40`.       |
| Llama 3.2 1B Instruct Q40         | 1.7 GB   | `python launch.py llama3_2_1b_instruct_q40`          |
| Llama 3.2 3B Instruct Q40         | 3.4 GB   | `python launch.py llama3_2_3b_instruct_q40`          |
| Llama 3.3 70B Instruct Q40        | 40 GB    | `python launch.py llama3_3_70b_instruct_q40`         |
| DeepSeek R1 Distill Llama 8B Q40  | 6.32 GB  | `python launch.py deepseek_r1_distill_llama_8b_q40`  |

### 🛠️ Convert Model Manually

Supported architectures: Llama.

* [How to Convert Llama 3.1](./docs/LLAMA.md)
* [How to Convert Hugging Face Model](./docs/HUGGINGFACE.md)

### 🚧 Known Limitations

* You can run Distributed Llama only on 1, 2, 4... 2^n nodes.
* The maximum number of nodes is equal to the number of KV heads in the model [#70](https://github.com/b4rtaz/distributed-llama/issues/70).
* Only the following quantizations are supported [#183](https://github.com/b4rtaz/distributed-llama/issues/183):
  * `q40` model with `q80` `buffer-float-type`
  * `f32` model with `f32` `buffer-float-type`

### 👷 Architecture

The project is split up into two parts:
* **Root node** - it's responsible for loading the model and weights and forward them to workers. Also, it synchronizes the state of the neural network. The root node is also a worker, it processes own slice of the neural network.
* **Worker node** - it processes own slice of the neural network. It doesn't require any configuration related to the model.

You always need the root node and you can add 2^n - 1 worker nodes to speed up the inference. The RAM usage of the neural network is split up across all nodes. The root node requires a bit more RAM than worker nodes.

### 🎹 Commands

* `dllama inference` - run the inference with a simple benchmark,
* `dllama chat` - run the CLI chat,
* `dllama worker` - run the worker node,
* `dllama-api` - run the API server.

<details>

<summary>🎹 Supported Arguments</summary>

<br />Inference, Chat, API

| Argument                     | Description                                                      | Example                                |
| ---------------------------- | ---------------------------------------------------------------- | -------------------------------------- |
| `--model <path>`             | Path to model.                                                   | `dllama_model_meta-llama-3-8b_q40.m`   |
| `--tokenizer <path>`         | Tokenizer to model.                                              | `dllama_tokenizer_llama3.t`            |
| `--buffer-float-type <type>` | Float precision of synchronization.                              | `q80`                                  |
| `--workers <workers>`        | Addresses of workers (ip:port), separated by space.              | `10.0.0.1:9999 10.0.0.2:9999`          |
| `--max-seq-len <n>`          | The maximum sequence length, it helps to reduce the RAM usage.   | `4096`                                 |

Inference, Chat, Worker, API

| Argument                     | Description                                                           | Example                             |
| ---------------------------- | --------------------------------------------------------------------- | ----------------------------------- |
| `--nthreads <n>`             | Amount of threads. Don't set a higher value than number of CPU cores. | `4`                                 |

Worker, API

| Argument                     | Description                       | Example           |
| ---------------------------- | --------------------------------- | ----------------- |
| `--port <port>`              | Binding port.                     | `9999`            |

Inference

| Argument                     | Description                    | Example            |
| ---------------------------- | ------------------------------ | ------------------ |
| `--prompt <prompt>`          | Initial prompt.                | `"Hello World"`    |
| `--steps <steps>`            | Number of tokens to generate.  | `256`              |

</details>

## 📊 Measurements

Please check the [discussions](https://github.com/b4rtaz/distributed-llama/discussions) section, where many measurements were published on different configurations.

## 🚀 Setup

Select and expand one of the sections below:

<details>

<summary>💻 MacOS, Linux, or Windows</summary>

<br />You need x86_64 AVX2 CPUs or ARM CPUs. Different devices may have different CPUs.

#### MacOS or Linux

The below instructions are for Debian-based distributions but you can easily adapt them to your distribution, macOS.

1. Install Git and GCC:
```sh
sudo apt install git build-essential
```
2. Clone this repository and compile Distributed Llama on all computers:
```sh
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
make dllama-api
```

Continue to point 3.

#### Windows

1. Install Git and Mingw (via [Chocolatey](https://chocolatey.org/install)):
```powershell
choco install mingw
```
2. Clone this repository and compile Distributed Llama on all computers:
```sh
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
make dllama-api
```

Continue to point 3.

#### Run Cluster

3. Transfer weights and the tokenizer file to the root computer.
4. Run worker nodes on worker computers:
```sh
./dllama worker --port 9999 --nthreads 4
```
5. Run root node on the root computer:
```sh
./dllama inference --model dllama_model_meta-llama-3-8b_q40.m --tokenizer dllama_tokenizer_llama3.t --buffer-float-type q80 --prompt "Hello world" --steps 16 --nthreads 4 --workers 192.168.0.1:9999
```

To add more worker nodes, just add more addresses to the `--workers` argument.

```
./dllama inference ... --workers 192.168.0.1:9999 192.168.0.2:9999 192.168.0.3:9999
```

</details>

<details>

<summary>📟 Raspberry Pi</summary>

<br />

1. Install `Raspberry Pi OS Lite (64 bit)` on your Raspberry Pi devices. This OS doesn't have desktop environment.
2. Connect all devices to your switch or router.
3. Connect to all devices via SSH.
```
ssh user@raspberrypi1.local
ssh user@raspberrypi2.local
```
4. Install Git:
```sh
sudo apt install git
```
5. Clone this repository and compile Distributed Llama on all devices:
```sh
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
make dllama-api
```
6. Transfer weights and the tokenizer file to the root device.
7. Optional: assign static IP addresses.
```sh
sudo ip addr add 10.0.0.1/24 dev eth0 # 1th device
sudo ip addr add 10.0.0.2/24 dev eth0 # 2th device
```
8. Run worker nodes on worker devices:
```sh
sudo nice -n -20 ./dllama worker --port 9999 --nthreads 4
```
9. Run root node on the root device:
```sh
sudo nice -n -20 ./dllama inference --model dllama_model_meta-llama-3-8b_q40.m --tokenizer dllama_tokenizer_llama3.t --buffer-float-type q80 --prompt "Hello world" --steps 16 --nthreads 4 --workers 10.0.0.2:9999
```

To add more worker nodes, just add more addresses to the `--workers` argument.

```
./dllama inference ... --workers 10.0.0.2:9999 10.0.0.3:9999 10.0.0.4:9999
```

</details>

## ✋ Contribution

Feel free to contribute to this project. For small changes, simply create a new merge request. For larger changes, please create an issue to discuss your plans. Please follow these guidelines when contributing:

* Make only minimal changes and avoid modifying files that are not necessary.
* Ensure the code is compatible across all supported systems and CPUs.
* This repository is maintained in English.

## 💡 License

This project is released under the MIT license.

## 📖 Citation

```
@misc{dllama,
  author = {Bartłomiej Tadych},
  title = {Distributed Llama},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/b4rtaz/distributed-llama}},
  commit = {7eb77ca93ec0d502e28d36b6fb20039b449cbea4}
}
```
