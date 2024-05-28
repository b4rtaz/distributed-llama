#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <cstdio>
#include "common/pthread.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define NEW_BUFFER(size) (char*)newBuffer(size)
#define FREE_BUFFER(buffer) freeBuffer(buffer)

void* newBuffer(size_t size);
void freeBuffer(void* buffer);

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
