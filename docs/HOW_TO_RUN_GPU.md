# How to Run Distributed Llama on 🧠 GPU

Distributed Llama can run on GPU devices using Vulkan API. This article describes how to build and run the project on GPU.

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

