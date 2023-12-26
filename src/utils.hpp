#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <pthread.h>

#define NEW_BUFFER(size) newBuffer(size)
#define FREE_BUFFER(buffer) free(buffer)

char* newBuffer(size_t size);
long timeMs();
unsigned int randomU32(unsigned long long *state);
float randomF32(unsigned long long *state);

#define TASK_LOOP_CONTINUE 0
#define TASK_LOOP_STOP -1

typedef int (*TaskLoopTask)(unsigned int threadIndex, void* userData);

struct TaskLoopState {
    unsigned int nThreads;
    unsigned int nTasks;
    TaskLoopTask* tasks;
    void* userData;
    std::atomic_uint currentTaskIndex;
    std::atomic_bool stop;
    std::atomic_uint doneThreadCount;
};

struct TaskLoopThread {
    unsigned int threadIndex;
    pthread_t handler;
    TaskLoopState* state;
};

class TaskLoop {
private:
    TaskLoopState state;
    TaskLoopThread* threads;
public:
    TaskLoop(unsigned int nThreads, unsigned int nTasks, TaskLoopTask* tasks, void* userData);
    ~TaskLoop();
    void run();
    static void* threadHandler(void* args);
};

#endif
