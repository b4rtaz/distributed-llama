#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include "utils.hpp"

#define BUFFER_ALIGNMENT 16

#ifdef _WIN32 
#include <windows.h>
#include <malloc.h>
#include <vector>
typedef DWORD thread_ret_t;

class LargeArrayManager {
public:
    LargeArrayManager(size_t size) {
        arr.reserve(size);
        arr.resize(size); // Allocate and resize to the desired size
    }

    char* getPointer() {
        return arr.data();
    }

private:
    std::vector<char> arr;
};

char* createBuffer(size_t size) {
    LargeArrayManager* manager = new LargeArrayManager(size);
    return manager->getPointer();
}

char* allocateAndLockMemory(size_t size) {
    char* buffer = createBuffer(size);
    std::memset(buffer, 0, size);
    return buffer;
}

static int check_align(size_t align)
{
    for (size_t i = sizeof(void *); i != 0; i *= 2)
    if (align == i)
        return 0;
    return EINVAL;
}

int posix_memalign(void **ptr, size_t align, size_t size) {
    if (check_align(align)) {
        fprintf(stderr, "posix_memalign: alignment %zu is not valid\n", align);
        return EINVAL;
    }

    int saved_errno = errno;
    void *p = _aligned_malloc(size, align);
    if (p == NULL) {
        fprintf(stderr, "posix_memalign: _aligned_malloc failed\n");
        errno = saved_errno;
        return ENOMEM;
    }

    *ptr = p;
    return 0;
}

#else
#include <sys/mman.h>

char* allocateAndLockMemory(size_t size) {
    char* ptr;

    if (posix_memalign((void**)&ptr, BUFFER_ALIGNMENT, size) != 0) {
        fprintf(stderr, "error: posix_memalign failed\n");
        exit(EXIT_FAILURE);
    }
    if (mlock(ptr, size) != 0) {
        fprintf(stderr, "🚧 Cannot allocate %zu bytes directly in RAM\n", size);
    }

    return ptr;
}
#endif

char* newBuffer(size_t size) {
    char* buffer = (char*)allocateAndLockMemory(size);
    if (buffer == NULL || buffer == nullptr) {
        fprintf(stderr, "🚧 Cannot allocate %zu bytes directly in RAM\n", size);
        exit(EXIT_FAILURE);
    }
    return buffer;
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
