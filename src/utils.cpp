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

long timeMs() {
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

TaskLoop::TaskLoop(unsigned int nThreads, unsigned int nTasks, TaskLoopTask* tasks, void* userData) {
    state.nThreads = nThreads;
    state.nTasks = nTasks;
    state.tasks = tasks;
    state.userData = userData;

    threads = new TaskLoopThread[nThreads];
    for (unsigned int i = 0; i < nThreads; i++) {
        threads[i].threadIndex = i;
        threads[i].state = &state;
    }
}

TaskLoop::~TaskLoop() {
    delete[] threads;
}

void TaskLoop::run() {
    state.currentTaskIndex.exchange(0);
    state.doneThreadCount.exchange(0);
    state.stop.exchange(false);

    for (unsigned int i = 1; i < state.nThreads; i++) {
        int result = pthread_create(&threads[i].handler, NULL, threadHandler, (void*)&threads[i]);
        if (result != 0) {
            printf("Cannot created thread\n");
            exit(EXIT_FAILURE);
        }
    }

    threadHandler((void*)&threads[0]);

    for (unsigned int i = 1; i < state.nThreads; i++) {
        pthread_join(threads[i].handler, NULL);
    }
}

void* TaskLoop::threadHandler(void* arg) {
    TaskLoopThread* context = (TaskLoopThread*)arg;
    TaskLoopState* state = context->state;
    unsigned int threadIndex = context->threadIndex;

    while (state->stop == false) {
        unsigned int currentTaskIndex = state->currentTaskIndex;

        int result = state->tasks[currentTaskIndex % state->nTasks](state->nThreads, threadIndex, state->userData);

        if (result == TASK_LOOP_STOP) {
            state->stop = true;
            break;
        }

        state->doneThreadCount++;

        if (threadIndex == 0) {
            while (state->stop == false && state->doneThreadCount < state->nThreads) {
                // NOP
            }
            state->doneThreadCount.exchange(0);
            state->currentTaskIndex.store(currentTaskIndex + 1);
        } else {
            while (state->stop == false && state->currentTaskIndex == currentTaskIndex) {
                // NOP
            }
        }
    }

    // printf("@ Thread %d stopped at step %d\n", threadIndex, unsigned(state->currentTaskIndex));
    return 0;
}
