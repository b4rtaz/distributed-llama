# How to Run Distributed Llama on 🧠 GPU

Distributed Llama can run on GPU devices using Vulkan API. This article describes how to build and run the project on GPU.

Before you start here, please check how to build and run Distributed Llama on CPU:
* [🍓 How to Run on Raspberry Pi](./HOW_TO_RUN_RASPBERRYPI.md)
* [💻 How to Run on Linux, MacOS or Windows](./HOW_TO_RUN_LINUX_MACOS_WIN.md)

To run on GPU, please follow these steps:

1. Install Vulkan SDK for your platform.
  * Linux: please check [this article](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html).
  * MacOS: download SDK [here](https://vulkan.lunarg.com/sdk/home#mac).
2. Build Distributed Llama with GPU support:

```bash
DLLAMA_VULKAN=1 make dllama
DLLAMA_VULKAN=1 make dllama-api
```

3. Now `dllama` and `dllama-api` binaries supports arguments related to GPU usage.

```
--gpu-index <index>   Use GPU device with given index (use `0` for first device)
```

4. You can run the root node or worker node on GPU by specifying the `--gpu-index` argument:

```bash
./dllama inference ... --gpu-index 0 
./dllama chat      ... --gpu-index 0 
./dllama worker    ... --gpu-index 0 
./dllama-api       ... --gpu-index 0 
```
