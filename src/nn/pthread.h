#ifndef PTHREAD_WRAPPER
#define PTHREAD_WRAPPER

#ifdef _WIN32 
#include <windows.h>

typedef HANDLE PthreadHandler;
typedef DWORD PthreadResult;
typedef DWORD (WINAPI *PthreadFunc)(void *);

static int pthread_create(PthreadHandler *out, void *unused, PthreadFunc func, void *arg) {
    (void) unused;
    PthreadHandler handle = CreateThread(NULL, 0, func, arg, 0, NULL);
    if (handle == NULL) {
        return EAGAIN;
    }
    *out = handle;
    return 0;
}

static int pthread_join(PthreadHandler thread, void *unused) {
    (void) unused;
    DWORD ret = WaitForSingleObject(thread, INFINITE);
    if (ret == WAIT_FAILED) {
        return -1;
    }
    CloseHandle(thread);
    return 0;
}
#else
#include <pthread.h>

typedef pthread_t PthreadHandler;
typedef void* PthreadResult;
typedef void* (*PthreadFunc)(void *);

#endif

#endif