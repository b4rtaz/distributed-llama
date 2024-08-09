# Distributed-Llama with Memory Budget

## Before my work...

First of all, I would like to thank b4rtaz for his work. His distributed-llama project has provided me with some new ideas and brought great convenience to my work.

>  https://github.com/b4rtaz/distributed-llama

However, if you have used b4rtaz's distributed-llama project, you will find that **<u>all nodes in its project will evenly distribute the amount of calculation (only for QKV calculation, forward propagation, activation function calculation, etc., excluding operations such as loading weights of root nodes).</u>**
For example, when spec->dim == 2048, if 4 nodes are used, each node needs to calculate 512 dimensions. But this is not the distributed result we want. **<u>We hope to make reasonable non-uniform distribution based on the memory, computing power, etc. of distributed devices.</u>** Based on this idea, I modified the source code and basically completed the above functions.

## New Features: Memory Budget

### How to use

In my fork, when you run dllama, you can add a new parameter named --memory-budget.

So your command could be:

For Root Node:

```cmd
./dllama inference --model PATH_TO_YOUR_MODEL --tokenizer PATH_TO_YOUR_TOKENIZER --buffer-float-type f32 --prompt "You are a person" --steps 16 --nthreads 4 --memory-budget 3 1 --workers x.x.x.x:9998
```

For Worker Node:

```cmd
./dllama worker --port 9998 --nthreads 4 --memory-budget 3 1
```

### about: --memory-budget

--memory-budget parameter accepts multiple int type inputs, the number of inputs should be the same as the number of summary Nodes. When you have 2 Nodes in total, you have to input --memory-budget with 2 int type, etc.

> :warning::warning::warning:
>
> Due to the existence of Kv shared header, there are restrictions on the division method according to parameters such as spec->dim and spec->headSize, as follows:
>
> 1. For TinyLlama 1.1B 3T Q40, it can be divided into 4 parts at most, and the 4 parts can be allocated arbitrarily, such as 3:1, 2:2, etc.;
> 2. For Llama 3 8B Q40, it can be divided into 8 parts at most, and the 8 parts can be allocated arbitrarily, such as 7:1, 3:2:2:1, 5:1:1:1, etc.
> 3. For Llama 3 70B Q40, Llama 3.1 405B Q40, etc., it should be possible to divide more, such as 16 parts, 32 parts, etc. However, due to hardware limitations, I did not try it. If you are interested, you can try more and welcome your feedback.
>
> :warning::warning::warning:

### Actual effect demonstration

![631dd69457d17baa4003d8fa5511e49e](C:\Users\yhbia\Documents\Tencent Files\2687952613\nt_qq\nt_data\Pic\2024-08\Ori\631dd69457d17baa4003d8fa5511e49e.png)

> Here I give a demonstration using Tiny Llama with a memory-budget of 3:1. I also tried using the Llama-3-8B model to perform an 8-partition and it worked.
>
> And of course, 12:4 will leads the same result with 3:1 :laughing:

## Some limitations and areas for improvement

**I must admit that there are still many loopholes in my project.**

1. I left many comments while reading the source code and debugging. I apologize if it brings inconvenience to your reading.
2. **<u>*This code currently only supports Linux system, and buffer-float-type only supports f32.*</u>**
  1. First, for Windows system, the send function of sockets.cpp, line 91 will return -1 at runtime, resulting in an error. I have not found a solution to this error. This error only occurs on Windows.
  2. Secondly, for other buffer-float-type, different forward functions need to be modified in func.cpp. The main modifications are the number of loop layers and assert part (due to the use of more detailed division, some assert functions for dimensions appear false, and the number of loop layers needs to be modified)

I will optimize these problems in the following period of time, and everyone is welcome to optimize together.

## Contact me

**If you have any problems or new ideas or anything else, welcome to contact me at: fromthefox@icloud.com or yhbian@std.uestc.edu.cn. Thx.**

