#pragma once
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

bool llamafile_sgemm(int m, int n, int k, const void *A, int lda, const void *B, int ldb, void *C,
                     int ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype);

#ifdef __cplusplus
}
#endif