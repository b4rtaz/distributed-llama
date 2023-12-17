#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include "shared-buffer.hpp"

SharedBuffer::SharedBuffer(size_t count) {
    this->count = count;
    slices = new size_t[count];
    bytes = new size_t[count];
    buffer = new char*[count];
}

SharedBuffer::~SharedBuffer() {
    for (int i = 0; i < count; i++) {
        delete[] buffer[i];
    }
    delete[] slices;
    delete[] bytes;
    delete[] buffer;
}

void SharedBuffer::createSliced(uint8_t bufferIndex, size_t bytes, size_t slices) {
    buffer[bufferIndex] = new char[bytes];
    this->bytes[bufferIndex] = bytes;
    this->slices[bufferIndex] = slices;
}

void SharedBuffer::createUnit(uint8_t bufferIndex, size_t bytes) {
    buffer[bufferIndex] = new char[bytes];
    this->bytes[bufferIndex] = bytes;
    this->slices[bufferIndex] = SLICES_UNIT;
}

char* SharedBuffer::getSliced(uint8_t bufferIndex, uint8_t sliceIndex) {
    size_t bytes = this->bytes[bufferIndex];
    size_t slices = this->slices[bufferIndex];
    if (slices == SLICES_UNIT) {
        printf("Buffer %hhu is not sliced\n", bufferIndex);
        exit(EXIT_FAILURE);
    }
    if (sliceIndex >= slices) {
        printf("Slice index %hhu out of range for buffer %hhu with %zu slices\n", sliceIndex, bufferIndex, slices);
        exit(EXIT_FAILURE);
    }
    size_t sliceOffset = bytes / slices;
    return buffer[bufferIndex] + sliceOffset * sliceIndex;
}

char* SharedBuffer::getUnit(uint8_t bufferIndex) {
    size_t slices = this->slices[bufferIndex];
    if (slices != SLICES_UNIT) {
        printf("Buffer %d is sliced\n", bufferIndex);
        exit(EXIT_FAILURE);
    }
    return buffer[bufferIndex];
}

size_t SharedBuffer::getSlices(uint8_t bufferIndex) {
    return slices[bufferIndex];
}

size_t SharedBuffer::getBytes(uint8_t bufferIndex) {
    return bytes[bufferIndex];
}
