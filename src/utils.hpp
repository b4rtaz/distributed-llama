#ifndef UTILS_HPP
#define UTILS_HPP
// ifndef --> if not define,用于防止重复定义头文件

#include <atomic>
// 支持原子操作
#include <cstdio>
#include "common/pthread.h"
// 多线程处理?

#ifdef _WIN32
#include <windows.h>
#endif
// 如果在_Windows平台上 --> include <windows.h>文件 相当于一个if

// 根据线程索引判断RANGE中的那个部分属于自己
#define SPLIT_RANGE_TO_THREADS(varStart, varEnd, rangeStart, rangeEnd, nThreads, threadIndex) \
    const unsigned int rangeLen = (rangeEnd - rangeStart); \
    const unsigned int rangeSlice = rangeLen / nThreads; \
    const unsigned int rangeRest = rangeLen % nThreads; \
    const unsigned int varStart = threadIndex * rangeSlice + (threadIndex < rangeRest ? threadIndex : rangeRest); \
    const unsigned int varEnd = varStart + rangeSlice + (threadIndex < rangeRest ? 1 : 0);
#if 0
varStart & varEnd:每个线程起始和结束的位置
rangeStart & rangeEnd:总范围的起始和结束的位置
nThreads:要分割的线程总数
threadIndex:线程索引
const unsigned int rangeSlice = rangeLen / nThreads; --> 将线程进行均分 --> 可以优化的地方?使用算法进行算力线性分配?
A?X:Y --> 三元操作符,如果A为真返回X,如果A为假返回Y;
rangeRest < nThreads,所以一定是部分Threads需要分配额外的Slice,这里选择的是按照Index分,但是实际上按照算力和Memory Budget分配更加合理?
#endif

#define DEBUG_FLOATS(name, v, n) printf("⭕ %s ", name); for (int i = 0; i < n; i++) printf("%f ", v[i]); printf("\n"); // DEBUG用的

void* newBuffer(size_t size); // 分配缓冲区
void freeBuffer(void* buffer); // 释放缓冲区

unsigned long timeMs(); // 获取当前时间
unsigned int randomU32(unsigned long long *state); // 随机生成无符号32位int
float randomF32(unsigned long long *state); // --> 随机生成32位float
long seekToEnd(FILE* file); // --> seekToEnd用于将指针移动到文件末尾,同时返回文件大小

struct MmapFile { // data + size + fd
    void* data;
    size_t size;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif
};

void openMmapFile(MmapFile* file, const char* path, size_t size);
void closeMmapFile(MmapFile* file); // 打开文件和关闭文件

typedef void (TaskLoopHandler)(unsigned int nThreads, unsigned int threadIndex, void* userData);
typedef struct { // --> 任务处理程序和任务类型
    TaskLoopHandler* handler;
    unsigned int taskType;
} TaskLoopTask;
// 

class TaskLoop;

struct TaskLoopThread {
    unsigned int threadIndex;
    unsigned int nTasks;
    dl_thread handler;
    TaskLoop* loop;
};

class TaskLoop {
public:
    unsigned int nThreads;
    unsigned int nTasks;
    unsigned int nTypes;
    TaskLoopTask* tasks;
    void* userData;
    std::atomic_uint currentTaskIndex;
    std::atomic_uint doneThreadCount;
    unsigned int lastTime;
    unsigned int* executionTime;
    TaskLoopThread* threads;

    TaskLoop(unsigned int nThreads, unsigned int nTasks, unsigned int nTypes, TaskLoopTask* tasks, void* userData);
    ~TaskLoop();
    
    // 每个线程对于每个任务,都会调用threadHandler函数,然后共同使得currentCount增加,增加到nThreads - 1时,说明对于这个任务,所有线程已经完成了任务,然后进行下一个任务;直到所有任务完成.推理完成.
    void run();
    static void* threadHandler(void* args);
};

#endif
