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
#include <malloc.h>
#else
#include <sys/mman.h>
#endif

void* gracefullyAllocateBuffer(size_t size){
    std::vector<char>* arr = new std::vector<char>();
    arr->reserve(size);
    arr->resize(size);
    char* buffer = arr->data();
    std::memset(buffer, 0, size);
    return buffer;
}

#ifdef _WIN32 
char* newBuffer(size_t size) {
    char* ptr = (char*)gracefullyAllocateBuffer(size);
    return ptr;
}
#else
char* newBuffer(size_t size) {
    char* ptr;
    bool useGraceful = false;

    if (posix_memalign((void**)&ptr, BUFFER_ALIGNMENT, size) != 0) {
        useGraceful = true;
    }
    else if (mlock(ptr, size) != 0){
        useGraceful = true;
    }

    if(useGraceful){
        ptr = (char*)gracefullyAllocateBuffer(size);
    }

    return ptr;
}
#endif

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

TaskLoop::TaskLoop(unsigned int nThreads, unsigned int nTasks, unsigned int nTypes, TaskLoopTask* tasks, void* userData) {
    this->nThreads = nThreads;
    this->nTasks = nTasks;
    this->nTypes = nTypes;
    this->tasks = tasks;
    this->userData = userData;
    executionTime = new unsigned int[nTypes];

    threads = new TaskLoopThread[nThreads];
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
    doneThreadCount.exchange(0);

    unsigned int i;
    lastTime = timeMs();
    for (i = 0; i < nTypes; i++) {
        executionTime[i] = 0;
    }

    for (i = 1; i < nThreads; i++) {
        int result = pthread_create(&threads[i].handler, NULL, (thread_func_t)threadHandler, (void*)&threads[i]);
        if (result != 0) {
            printf("Cannot created thread\n");
            exit(EXIT_FAILURE);
        }
    }

    threadHandler((void*)&threads[0]);

    for (i = 1; i < nThreads; i++) {
        pthread_join(threads[i].handler, NULL);
    }
}

void* TaskLoop::threadHandler(void* arg) {
    TaskLoopThread* context = (TaskLoopThread*)arg;
    TaskLoop* loop = context->loop;
    unsigned int threadIndex = context->threadIndex;

    while (true) {
        const unsigned int currentTaskIndex = loop->currentTaskIndex.load();
        if (currentTaskIndex == context->nTasks) {
            break;
        }

        const TaskLoopTask* task = &loop->tasks[currentTaskIndex % loop->nTasks];

        task->handler(loop->nThreads, threadIndex, loop->userData);

        int currentCount = loop->doneThreadCount.fetch_add(1);

        if (currentCount == loop->nThreads - 1) {
            unsigned int currentTime = timeMs();
            loop->executionTime[task->taskType] += currentTime - loop->lastTime;
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
