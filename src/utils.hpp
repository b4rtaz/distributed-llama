#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <pthread.h>

#define NEW_BUFFER(size) newBuffer(size)
#define FREE_BUFFER(buffer) free(buffer)

char* newBuffer(size_t size);
unsigned long timeMs();
unsigned int randomU32(unsigned long long *state);
float randomF32(unsigned long long *state);

typedef void (TaskLoopHandler)(unsigned int nThreads, unsigned int threadIndex, void* userData);
typedef struct {
    TaskLoopHandler* handler;
    unsigned int taskType;
} TaskLoopTask;

class TaskLoop;

struct TaskLoopThread {
    unsigned int threadIndex;
    unsigned int nTasks;
    pthread_t handler;
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
