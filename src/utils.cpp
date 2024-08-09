#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "utils.hpp"

#define BUFFER_ALIGNMENT 16

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

void* newBuffer(size_t size) {
    void* buffer;
#ifdef _WIN32
    buffer = _aligned_malloc(size, BUFFER_ALIGNMENT);
    if (buffer == NULL) {
        fprintf(stderr, "error: _aligned_malloc failed\n");
        exit(EXIT_FAILURE);
    }
#else
    if (posix_memalign((void**)&buffer, BUFFER_ALIGNMENT, size) != 0) {
        fprintf(stderr, "error: posix_memalign failed\n");
        exit(EXIT_FAILURE);
    }
    if (mlock(buffer, size) != 0) {
        fprintf(stderr, "🚧 Cannot allocate %zu bytes directly in RAM\n", size);
    }
#endif
    return buffer;
}

void freeBuffer(void* buffer) {
#ifdef _WIN32
    _aligned_free(buffer);
#else
    free(buffer);
#endif
}

unsigned long timeMs() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000LL + te.tv_usec / 1000;
}

unsigned int randomU32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float randomF32(unsigned long long *state) {
    // random float32 in <0,1)
    return (randomU32(state) >> 8) / 16777216.0f;
}

long seekToEnd(FILE* file) { // --> seekToEnd用于将指针移动到文件末尾,同时返回文件大小
#ifdef _WIN32
    _fseeki64(file, 0, SEEK_END);
    return _ftelli64(file);
#else
    fseek(file, 0, SEEK_END);
    return ftell(file);
#endif
}

void openMmapFile(MmapFile* file, const char* path, size_t size) {
    file->size = size;
#ifdef _WIN32
    file->hFile = CreateFileA(path, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file->hFile == INVALID_HANDLE_VALUE) {
        printf("Cannot open file %s\n", path);
        exit(EXIT_FAILURE);
    }

    file->hMapping = CreateFileMappingA(file->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file->hMapping == NULL) {
        printf("CreateFileMappingA failed, error: %lu\n", GetLastError());
        CloseHandle(file->hFile);
        exit(EXIT_FAILURE);
    }

    file->data = (char*)MapViewOfFile(file->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (file->data == NULL) {
        printf("MapViewOfFile failed!\n");
        CloseHandle(file->hMapping);
        CloseHandle(file->hFile);
        exit(EXIT_FAILURE);
    }
#else
    file->fd = open(path, O_RDONLY);
    if (file->fd == -1) {
        printf("Cannot open file %s\n", path);
        exit(EXIT_FAILURE);
    }

    file->data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, file->fd, 0);
    if (file->data == MAP_FAILED) {
        printf("Mmap failed!\n");
        close(file->fd);
        exit(EXIT_FAILURE);
    }
#endif
}

void closeMmapFile(MmapFile* file) {
#ifdef _WIN32
    UnmapViewOfFile(file->data);
    CloseHandle(file->hMapping);
    CloseHandle(file->hFile);
#else
    munmap(file->data, file->size);
    close(file->fd);
#endif
}

TaskLoop::TaskLoop(unsigned int nThreads, unsigned int nTasks, unsigned int nTypes, TaskLoopTask* tasks, void* userData) {
    this->nThreads = nThreads;
    this->nTasks = nTasks;
    this->nTypes = nTypes;
    this->tasks = tasks;
    this->userData = userData;
    executionTime = new unsigned int[nTypes]; // nTypes == 2 --> Transfer | Inference

    threads = new TaskLoopThread[nThreads];
    /*
    struct TaskLoopThread {
    unsigned int threadIndex;
    unsigned int nTasks;
    dl_thread handler;
    TaskLoop* loop;
    };
    */
    for (unsigned int i = 0; i < nThreads; i++) {
        threads[i].threadIndex = i;
        threads[i].nTasks = nTasks;
        threads[i].loop = this;
    }
}

TaskLoop::~TaskLoop() {
    delete[] executionTime;
    delete[] threads;
}

void TaskLoop::run() {
    currentTaskIndex.exchange(0);
    doneThreadCount.exchange(0); // 将两个变量设置为0

    unsigned int i;
    lastTime = timeMs();
    for (i = 0; i < nTypes; i++) {
        executionTime[i] = 0;
    }
    for (i = 1; i < nThreads; i++) { // 从1开始,因为线程0由主线程直接处理
        int result = pthread_create(&threads[i].handler, NULL, (thread_func_t)threadHandler, (void*)&threads[i]); // 每个线程都会执行threadHandler函数
        // 将handler存储到&threads[i].handler中
        // 创建线程,创建成功返回0
        if (result != 0) {
            printf("Cannot created thread\n");
            exit(EXIT_FAILURE);
        }
    }
    // HERE !!!!
    threadHandler((void*)&threads[0]); // 线程0由主线程处理
    for (i = 1; i < nThreads; i++) {
        pthread_join(threads[i].handler, NULL); // FOR循环,等待其他线程完成
    }
}

void* TaskLoop::threadHandler(void* arg) {
    TaskLoopThread* context = (TaskLoopThread*)arg;
    TaskLoop* loop = context->loop;
    unsigned int threadIndex = context->threadIndex; // 不就是0?
    int i = 0;
    while (true) { // 每个线程对于每个任务,都会调用threadHandler函数,然后共同使得currentCount增加,增加到nThreads - 1时,说明对于这个任务,所有线程已经完成了任务,然后进行下一个任务;直到所有任务完成.推理完成.
        i++;
        const unsigned int currentTaskIndex = loop->currentTaskIndex.load();
        // printf("\n currentTaskIndex %d\n", currentTaskIndex);
        // printf("\n context->nTasks %d\n", context->nTasks);
        if (currentTaskIndex == context->nTasks) { // --> 已经完成所有Task // 正在处理的TaskIndex == 总的任务数
            break;
        }
        // printf("\n currentTaskIndex %d\n", currentTaskIndex);
        const TaskLoopTask* task = &loop->tasks[currentTaskIndex % loop->nTasks]; // 获取当前任务的指针
        // printf("\n task pointer %p\n", task);
        /*
        typedef struct {
            TaskLoopHandler* handler;
            unsigned int taskType;
        } TaskLoopTask;
        */
        // 目前的猜测是task指针的问题
        // printf("\n task->handler %p\n", task->handler);
        task->handler(loop->nThreads, threadIndex, loop->userData); // 执行任务?奇怪!
        int currentCount = loop->doneThreadCount.fetch_add(1); // 已完成的线程数量+1

        if (currentCount == loop->nThreads - 1) { // 所有线程都已经完成了任务
            unsigned int currentTime = timeMs();
            loop->executionTime[task->taskType] += currentTime - loop->lastTime; // task->taskType = Transfer | Inference
            loop->lastTime = currentTime;

            loop->doneThreadCount.store(0);
            loop->currentTaskIndex.fetch_add(1);
        } else {
            while (loop->currentTaskIndex.load() == currentTaskIndex) {
                // NOP
            }
        }
    }

    // printf("@ Thread %d stopped at step %d\n", threadIndex, unsigned(loop->currentTaskIndex));
    return 0;
}
