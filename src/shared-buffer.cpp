#include "shared-buffer.hpp"
#include <cstring>
#include <stdlib.h>
#include <stdio.h>

SharedBuffer::SharedBuffer(int count) {
    this->count = count;
    buffer = new char*[count];
    startSliceIndex = new int[count];
    sliceSize = new int[count];
}

SharedBuffer::~SharedBuffer() {
    for (int i = 0; i < count; i++) {
        delete[] buffer[i];
    }
    delete[] buffer;
    delete[] startSliceIndex;
    delete[] sliceSize;
}

void SharedBuffer::create(int bufferIndex, int startSliceIndex, int endSliceIndex, int sliceSize) {
    int count = endSliceIndex - startSliceIndex;
    buffer[bufferIndex] = new char[count * sliceSize];
    this->startSliceIndex[bufferIndex] = startSliceIndex;
    this->sliceSize[bufferIndex] = sliceSize;
}

char* SharedBuffer::get(int bufferIndex, int sliceIndex) {
    int index = startSliceIndex[bufferIndex] + sliceIndex;
    if (index < 0) {
        printf("Invalid index\n");
        exit(1);
    }
    return buffer[bufferIndex] + sliceSize[bufferIndex] * index;
}

void SharedBuffer::sync(int bufferIndex, int sliceIndex) {
    // TODO
}
