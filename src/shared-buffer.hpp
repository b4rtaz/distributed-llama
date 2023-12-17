#include <cstdlib>

#ifndef shared_buffer_hpp
#define shared_buffer_hpp

#define SLICES_UNIT 9999

class SharedBuffer {
private:
    size_t count;
    size_t* slices;
    size_t* bytes;
    char** buffer;
public:
    SharedBuffer(size_t count);
    ~SharedBuffer();
    void createSliced(uint8_t bufferIndex, size_t bytes, size_t slices);
    void createUnit(uint8_t bufferIndex, size_t bytes);
    char* getSliced(uint8_t bufferIndex, uint8_t sliceIndex);
    char* getUnit(uint8_t bufferIndex);
    size_t getSlices(uint8_t bufferIndex);
    size_t getBytes(uint8_t bufferIndex);
};

#endif
