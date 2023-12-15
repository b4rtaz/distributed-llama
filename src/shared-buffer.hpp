#include <stdint.h>

#ifndef shared_buffer_hpp
#define shared_buffer_hpp

#define SLICES_UNIT -1

class SharedBuffer {
private:
    int count;
    int* slices;
    int* bytes;
    char** buffer;
public:
    SharedBuffer(int count);
    ~SharedBuffer();
    void createSliced(uint8_t bufferIndex, int bytes, int slices);
    void createUnit(uint8_t bufferIndex, int bytes);
    char* getSliced(uint8_t bufferIndex, uint8_t sliceIndex);
    char* getUnit(uint8_t bufferIndex);
    int getSlices(uint8_t bufferIndex);
    int getBytes(uint8_t bufferIndex);
};

#endif
