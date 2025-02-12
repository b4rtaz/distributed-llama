#ifndef MMAP_HPP
#define MMAP_HPP

#include <cstdio>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

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

long seekToEnd(FILE* file) {
#ifdef _WIN32
    _fseeki64(file, 0, SEEK_END);
    return _ftelli64(file);
#else
    fseek(file, 0, SEEK_END);
    return ftell(file);
#endif
}

void openMmapFile(MmapFile *file, const char *path, size_t size) {
    file->size = size;
#ifdef _WIN32
    file->hFile = CreateFileA(path, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file->hFile == INVALID_HANDLE_VALUE) {
        printf("Cannot open file %s\n", path);
        exit(EXIT_FAILURE);
    }

    file->hMapping = CreateFileMappingA(file->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file->hMapping == NULL) {
        printf("CreateFileMappingA failed, error: %lu\n", GetLastError());
        CloseHandle(file->hFile);
        exit(EXIT_FAILURE);
    }

    file->data = (void *)MapViewOfFile(file->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (file->data == NULL) {
        printf("MapViewOfFile failed!\n");
        CloseHandle(file->hMapping);
        CloseHandle(file->hFile);
        exit(EXIT_FAILURE);
    }
#else
    file->fd = open(path, O_RDONLY);
    if (file->fd == -1) {
        throw std::runtime_error("Cannot open file");
    }

    file->data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, file->fd, 0);
    if (file->data == MAP_FAILED) {
        close(file->fd);
        throw std::runtime_error("Mmap failed");
    }
#endif
}

void closeMmapFile(MmapFile *file) {
#ifdef _WIN32
    UnmapViewOfFile(file->data);
    CloseHandle(file->hMapping);
    CloseHandle(file->hFile);
#else
    munmap(file->data, file->size);
    close(file->fd);
#endif
}

#endif