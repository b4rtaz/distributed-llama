#ifndef NN_VULKAN_HPP
#define NN_VULKAN_HPP

#include <vulkan/vulkan.hpp>
#include <vector>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

#define DEBUG_VULKAN_TRACE false

typedef struct {
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    uint32_t queueFamilyIndex;
    vk::CommandPool commandPool;
    vk::Queue queue;
} NnVulkanContext;

enum NnStagingVulkanCopyDirection {
    COPY_TO_DEVICE,
    COPY_FROM_DEVICE
};

class NnVulkanStagingCopy {
private:
    NnStagingVulkanCopyDirection direction;
    const NnVulkanContext *context;
    vk::DeviceSize bufferSize;
    vk::Buffer deviceBuffer;
    vk::Buffer hostBuffer;
    vk::DeviceMemory hostMemory;
    void *hostPointer;
public:
    NnVulkanStagingCopy(const NnVulkanContext *context, vk::Buffer& deviceBuffer, const vk::DeviceSize bufferSize, const NnStagingVulkanCopyDirection direction);
    ~NnVulkanStagingCopy();
    void copy(NnByte *data);
    void executeCopyCommand();
    void addCopyCommand(vk::CommandBuffer& commandBuffer);
};

class NnVulkanBuffer {
private:
    bool isHostVisible;
    NnVulkanContext *context;
    vk::DeviceMemory deviceMemory;
    void *hostPointer;
public:
    vk::DeviceSize bufferSize;
    vk::Buffer deviceBuffer;
    vk::BufferUsageFlags usageFlags;
    NnVulkanBuffer(NnVulkanContext *context, const vk::DeviceSize bufferSize, vk::BufferUsageFlags usageFlags, bool fastAccess);
    ~NnVulkanBuffer();
    void write(const NnByte *data);
    void read(NnByte *data);
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
    NnVulkanDeviceData(NnVulkanContext *context, NnNetConfig *netConfig, NnNodeConfig *nodeConfig);
    ~NnVulkanDeviceData();

    NnSize2D resolveBufferSize(NnPointerConfig *config);
    NnVulkanBuffer *resolvePointerVulkanBuffer(NnPointerConfig *config);
    NnUint resolveBufferBatchOffset(NnPointerConfig *config, NnUint batchIndex);
    NnUint resolveBufferBatchWidth(NnPointerConfig *config, NnUint batchIndex);
};

class NnVulkanDevice : public NnDevice {
private:
    NnVulkanContext context;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
public:
    NnVulkanDeviceData *data;
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
    NnVulkanDeviceSegmentData(NnVulkanContext *context, NnVulkanDeviceData *data, NnSegmentConfig *segmentConfig, NnUint nBatches);
    NnVulkanBuffer *resolveOpBatchInfoVulkanBuffer(NnUint opIndex);
    NnVulkanBuffer *resolveOpWeightVulkanBuffer(NnUint opIndex);
    NnVulkanBuffer *resolveOpConfigVulkanBuffer(NnUint opIndex);
};

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
    NnUint lastBatchSize;
public:
    NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution);
    ~NnVulkanDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif