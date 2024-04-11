#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <sys/mman.h>
#include "utils.hpp"

#define BUFFER_ALIGNMENT 16

char* newBuffer(size_t size) {
    char* buffer;
    if (posix_memalign((void**)&buffer, BUFFER_ALIGNMENT, size) != 0) {
        fprintf(stderr, "error: posix_memalign failed\n");
        exit(EXIT_FAILURE);
    }
    if (mlock(buffer, size) != 0) {
        fprintf(stderr, "🚧 Cannot allocate %zu bytes directly in RAM\n", size);
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
        threads[i].loop = this;
    }
}

TaskLoop::~TaskLoop() {
    delete[] executionTime;
    delete[] threads;
}

void TaskLoop::run() {
    currentTaskIndex.exchange(0);
    stopTaskIndex.exchange(0);
    doneThreadCount.exchange(0);

    unsigned int i;
    lastTime = timeMs();
    for (i = 0; i < nTypes; i++) {
        executionTime[i] = 0;
    }

    for (i = 1; i < nThreads; i++) {
        int result = pthread_create(&threads[i].handler, NULL, threadHandler, (void*)&threads[i]);
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
        if (currentTaskIndex != 0 && currentTaskIndex == loop->stopTaskIndex.load()) {
            break;
        }

        const TaskLoopTask* task = &loop->tasks[currentTaskIndex % loop->nTasks];

        int result = task->handler(loop->nThreads, threadIndex, loop->userData);

        if (result == TASK_STOP) {
            loop->stopTaskIndex.store(currentTaskIndex + 1);
        }

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

    // printf("@ Thread %d stopped at step %d\n", threadIndex, unsigned(state->currentTaskIndex));
    return 0;
}
