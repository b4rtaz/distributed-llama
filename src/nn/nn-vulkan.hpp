#ifndef NN_VULKAN_HPP
#define NN_VULKAN_HPP

#include <vulkan/vulkan.hpp>
#include <vector>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

class NnVulkanContext {
public:
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    uint32_t queueFamilyIndex;
    vk::CommandPool commandPool;
    vk::Queue queue;
    NnSize nonCoherentAtomSize;
    NnVulkanContext(const NnUint gpuIndex);
    ~NnVulkanContext();
    std::pair<vk::Buffer, vk::DeviceMemory> createRawBuffer(const uint32_t memoryTypeIndex, const vk::DeviceSize bufferSize, const vk::BufferUsageFlags usageFlags);
};

enum NnVulkanStagingCopierDirection {
    COPY_TO_DEVICE,
    COPY_FROM_DEVICE
};

class NnVulkanStagingCopier {
private:
    NnVulkanContext *context;
    uint32_t memoryTypeIndex;

    vk::DeviceSize allocatedSize;
    vk::Buffer hostBuffer;
    vk::DeviceMemory hostMemory;
    void *hostPointer;
public:
    NnVulkanStagingCopier(NnVulkanContext *context);
    ~NnVulkanStagingCopier();
    void allocate(const NnSize size);
    void copy(NnByte *data, const NnSize size, const NnVulkanStagingCopierDirection direction);
    void executeCopyCommand(vk::Buffer& target, const NnSize offset, const NnSize size, const NnVulkanStagingCopierDirection direction);
    void addCopyCommand(vk::CommandBuffer& commandBuffer, vk::Buffer& target, const NnSize offset, const NnSize size, const NnVulkanStagingCopierDirection direction);
    void tryRelease();
};

class NnVulkanBuffer {
private:
    bool isHostVisible;
    NnVulkanContext *context;
    NnVulkanStagingCopier *copier;
    vk::DeviceMemory deviceMemory;
    NnByte *hostPointer;
public:
    const char *name;
    NnSize bufferSize;
    bool isSliceable;
    vk::Buffer deviceBuffer;
    vk::BufferUsageFlags usageFlags;
    NnVulkanBuffer(NnVulkanContext *context, NnVulkanStagingCopier *copier, const char *name, const NnSize bufferSize, const bool isSliceable, vk::BufferUsageFlags usageFlags, bool fastAccess);
    ~NnVulkanBuffer();
    void write(const NnByte *data);
    void write(const NnByte *data, const NnSize offset, const NnSize size);
    void read(NnByte *data);
    void read(NnByte *data, const NnSize offset, const NnSize size);
    NnSize calcSliceSize(const NnSize nominator, const NnSize denominator);
};

class NnVulkanBufferFactory {
private:
    NnVulkanContext *context;
    NnVulkanStagingCopier *copier;
public:
    NnVulkanBufferFactory(NnVulkanContext *context, NnVulkanStagingCopier *copier);
    std::unique_ptr<NnVulkanBuffer> create(const char *name, const NnSize bufferSize, const bool isSliceable, vk::BufferUsageFlags usageFlags, bool fastAccess);
};

typedef struct {
    NnUint inputOffset;
    NnUint inputSizeX;
    NnUint outputOffset;
    NnUint outputSizeX;
} NnVulkanBatchInfo;

class NnVulkanDeviceData {
private:
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
public:
    std::vector<std::unique_ptr<NnVulkanBuffer>> pipes;
    std::vector<std::unique_ptr<NnVulkanBuffer>> buffers;
    std::vector<std::unique_ptr<NnVulkanBuffer>> internalBuffers;
    NnVulkanDeviceData(NnVulkanBufferFactory *bufferFactory, NnNetConfig *netConfig, NnNodeConfig *nodeConfig);
    ~NnVulkanDeviceData();

    NnSize3D resolveBufferSize(NnPointerConfig *config);
    NnVulkanBuffer *resolvePointerVulkanBuffer(NnPointerConfig *config);
    NnUint resolveBufferBatchOffset(NnPointerConfig *config, NnUint batchIndex, NnUint zIndex);
    NnUint resolveBufferBatchWidth(NnPointerConfig *config);
    NnVulkanBuffer *resolvePipeByIndex(NnUint pipeIndex);
    NnVulkanBuffer *resolveBufferByIndex(NnUint bufferIndex);
};

class NnVulkanDevice : public NnDevice {
private:
    NnVulkanContext context;
    NnVulkanStagingCopier copier;
    NnVulkanBufferFactory bufferFactory;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
public:
    NnVulkanDeviceData data;
    NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnVulkanDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
};

class NnVulkanDeviceSegmentData {
private:
    NnVulkanDeviceData *data;
    std::vector<NnUint> batchInfoBufferIndex;
    std::vector<NnUint> weightBufferIndex;
    std::vector<NnUint> configBufferIndex;
public:
    NnVulkanDeviceSegmentData(NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnSegmentConfig *segmentConfig, NnUint nBatches);
    NnVulkanBuffer *resolveOpBatchInfoVulkanBuffer(NnUint opIndex);
    NnVulkanBuffer *resolveOpWeightVulkanBuffer(NnUint opIndex);
    NnVulkanBuffer *resolveOpConfigVulkanBuffer(NnUint opIndex);
};

enum NnOpBufferAccessType {
    ACCESS_IMMUTABLE,
    ACCESS_READONLY,
    ACCESS_READ_WRITE,
};

typedef struct {
    NnOpBufferAccessType type;
    NnVulkanBuffer *buffer;
} NnOpBufferAccess;

class NnVulkanDeviceSegment : public NnDeviceSegment {
private:
    NnVulkanContext *context;
    NnVulkanDeviceData *data;
    NnNetConfig *netConfig;
    NnUint segmentIndex;
    NnSegmentConfig *segmentConfig;
    NnNetExecution *netExecution;
    std::unique_ptr<NnVulkanDeviceSegmentData> segmentData;

    std::vector<vk::ShaderModule> shaderModules;
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::Fence fence;
    std::vector<vk::PipelineLayout> pipelineLayouts;
    std::vector<vk::Pipeline> pipelines;
    vk::PipelineCache pipelineCache;
    vk::CommandBuffer commandBuffer;
    std::vector<std::vector<NnVulkanBuffer *>> buffersToSync;
    NnUint lastBatchSize;
public:
    NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanBufferFactory *bufferFactory, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution);
    ~NnVulkanDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif