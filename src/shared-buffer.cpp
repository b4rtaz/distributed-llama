#include "shared-buffer.hpp"
#include <cstring>
#include <stdlib.h>
#include <stdio.h>

SharedBuffer::SharedBuffer(int count) {
    this->count = count;
    slices = new int[count];
    bytes = new int[count];
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

void SharedBuffer::createSliced(uint8_t bufferIndex, int bytes, int slices) {
    buffer[bufferIndex] = new char[bytes];
    this->bytes[bufferIndex] = bytes;
    this->slices[bufferIndex] = slices;
}

void SharedBuffer::createUnit(uint8_t bufferIndex, int bytes) {
    buffer[bufferIndex] = new char[bytes];
    this->bytes[bufferIndex] = bytes;
    this->slices[bufferIndex] = SLICES_UNIT;
}

char* SharedBuffer::getSliced(uint8_t bufferIndex, uint8_t sliceIndex) {
    int bytes = this->bytes[bufferIndex];
    int slices = this->slices[bufferIndex];
    if (slices == SLICES_UNIT) {
        printf("Buffer %d is not sliced\n", bufferIndex);
        exit(EXIT_FAILURE);
    }
    if (sliceIndex >= slices) {
        printf("Slice index %d out of range for buffer %d with %d slices\n", sliceIndex, bufferIndex, slices);
        exit(EXIT_FAILURE);
    }
    int sliceOffset = bytes / slices;
    return buffer[bufferIndex] + sliceOffset * sliceIndex;
}

char* SharedBuffer::getUnit(uint8_t bufferIndex) {
    int slices = this->slices[bufferIndex];
    if (slices != SLICES_UNIT) {
        printf("Buffer %d is sliced\n", bufferIndex);
        exit(EXIT_FAILURE);
    }
    return buffer[bufferIndex];
}

int SharedBuffer::getSlices(uint8_t bufferIndex) {
    return slices[bufferIndex];
}

int SharedBuffer::getBytes(uint8_t bufferIndex) {
    return bytes[bufferIndex];
}
