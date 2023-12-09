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

void SharedBuffer::createSliced(int bufferIndex, int bytes, int slices) {
    buffer[bufferIndex] = new char[bytes];
    this->bytes[bufferIndex] = bytes;
    this->slices[bufferIndex] = slices;
}

void SharedBuffer::createUnit(int bufferIndex, int bytes) {
    buffer[bufferIndex] = new char[bytes];
    this->bytes[bufferIndex] = bytes;
    this->slices[bufferIndex] = -1;
}

char* SharedBuffer::getSliced(int bufferIndex, int sliceIndex) {
    int bytes = this->bytes[bufferIndex];
    int slices = this->slices[bufferIndex];
    if (slices == -1) {
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

char* SharedBuffer::getUnit(int bufferIndex) {
    int slices = this->slices[bufferIndex];
    if (slices != -1) {
        printf("Buffer %d is sliced\n", bufferIndex);
        exit(EXIT_FAILURE);
    }
    return buffer[bufferIndex];
}

void SharedBuffer::send(int bufferIndex) {
    // TODO
}
