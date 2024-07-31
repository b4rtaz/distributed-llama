#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <cstdio>
#include "common/pthread.h"

#define ALLOC_MEMORY true

#ifdef _WIN32
#include <windows.h>
#endif

#define SPLIT_RANGE_TO_THREADS(varStart, varEnd, rangeStart, rangeEnd, nThreads, threadIndex) \
    const unsigned int rangeLen = (rangeEnd - rangeStart); \
    const unsigned int rangeSlice = rangeLen / nThreads; \
    const unsigned int rangeRest = rangeLen % nThreads; \
    const unsigned int varStart = threadIndex * rangeSlice + (threadIndex < rangeRest ? threadIndex : rangeRest); \
    const unsigned int varEnd = varStart + rangeSlice + (threadIndex < rangeRest ? 1 : 0);

#define DEBUG_FLOATS(name, v, n) printf("⭕ %s ", name); for (int i = 0; i < n; i++) printf("%f ", v[i]); printf("\n");

void* newBuffer(size_t size);
void freeBuffer(void* buffer);

void* newMmapFileBuffer(unsigned int appInstanceId, size_t size);
void freeMmapFileBuffer(void* addr);

unsigned long timeMs();
unsigned int randomU32(unsigned long long *state);
float randomF32(unsigned long long *state);
long seekToEnd(FILE* file);

struct MmapFile {
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
void closeMmapFile(MmapFile* file);

typedef void (TaskLoopHandler)(unsigned int nThreads, unsigned int threadIndex, void* userData);
typedef struct {
    TaskLoopHandler* handler;
    unsigned int taskType;
} TaskLoopTask;

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
    void run();
    static void* threadHandler(void* args);
};

#endif
