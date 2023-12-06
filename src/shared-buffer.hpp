#ifndef shared_buffer_hpp
#define shared_buffer_hpp

class SharedBuffer {
private:
    int count;
    char** buffer;
    int* startSliceIndex;
    int* sliceSize;
public:
    SharedBuffer(int count);
    ~SharedBuffer();
    void create(int bufferIndex, int startSliceIndex, int endSliceIndex, int sliceSize);
    char* get(int bufferIndex, int sliceIndex);
    void sync(int bufferIndex, int sliceIndex);
};

#endif
