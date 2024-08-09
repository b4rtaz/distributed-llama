#ifndef PTHREAD_WRAPPER
#define PTHREAD_WRAPPER

#ifdef _WIN32 
#include <windows.h>

typedef HANDLE dl_thread;
typedef DWORD thread_ret_t;
typedef DWORD (WINAPI *thread_func_t)(void *);

static int pthread_create(dl_thread * out, void * unused, thread_func_t func, void * arg) {
    (void) unused;
    dl_thread handle = CreateThread(NULL, 0, func, arg, 0, NULL);// 在Windows平台上创建线程,执行方法就是threadHandler方法,也就是这里传入的func
    if (handle == NULL) {// 如果创建失败
        return EAGAIN;
    }

    *out = handle; // 将线程存储
    return 0;
}

static int pthread_join(dl_thread thread, void * unused) {
    (void) unused;
    DWORD ret = WaitForSingleObject(thread, INFINITE); // INFINITE无线等待线程创建,直到线程终止
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
