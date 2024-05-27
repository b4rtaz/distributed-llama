#ifndef PTHREAD_WRAPPER
#define PTHREAD_WRAPPER

#ifdef _WIN32 
#include <windows.h>

typedef HANDLE dl_thread;
typedef DWORD thread_ret_t;
typedef DWORD (WINAPI *thread_func_t)(void *);

static int pthread_create(dl_thread * out, void * unused, thread_func_t func, void * arg) {
    (void) unused;
    dl_thread handle = CreateThread(NULL, 0, func, arg, 0, NULL);
    if (handle == NULL) {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(dl_thread thread, void * unused) {
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

typedef pthread_t dl_thread;
typedef void* thread_ret_t;
typedef void* (*thread_func_t)(void *);

#endif

#endif  // PTHREAD_WRAPPER
