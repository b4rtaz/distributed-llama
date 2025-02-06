#ifndef LLAMAFILE_SGEMM_H
#define LLAMAFILE_SGEMM_H

#include <cstdint>

bool llamafile_sgemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda, const void *B, int64_t ldb, void *C,
    int64_t ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype);

#endif
