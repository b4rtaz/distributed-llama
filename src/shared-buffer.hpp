#ifndef shared_buffer_hpp
#define shared_buffer_hpp

class SharedBuffer {
private:
    int count;
    int* slices;
    int* bytes;
    char** buffer;
public:
    SharedBuffer(int count);
    ~SharedBuffer();
    void createSliced(int bufferIndex, int bytes, int slices);
    void createUnit(int bufferIndex, int bytes);
    char* getSliced(int bufferIndex, int sliceIndex);
    char* getUnit(int bufferIndex);
    void send(int bufferIndex);
};

#endif
