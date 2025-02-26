#ifndef NN_VULKAN_HPP
#define NN_VULKAN_HPP

#include <vulkan/vulkan.hpp>
#include <vector>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

#define DEBUG_VULKAN_TRACE true

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
    void addCopyCommand(vk::CommandBuffer commandBuffer);
};

class NnVulkanBuffer {
private:
    bool isHostVisible;
    NnVulkanContext *context;
    vk::DeviceSize bufferSize;
    vk::Buffer deviceBuffer;
    vk::DeviceMemory deviceMemory;
    void *hostPointer;
public:
    NnVulkanBuffer(NnVulkanContext *context, const vk::DeviceSize bufferSize, vk::BufferUsageFlags usageFlags, bool fastAccess);
    ~NnVulkanBuffer();
    void write(const NnByte *data);
};

class NnVulkanData {
public:
    std::vector<std::unique_ptr<NnVulkanBuffer>> pipes;
    std::vector<std::unique_ptr<NnVulkanBuffer>> buffers;
    NnVulkanData(const NnUint nPipes, const NnUint nBuffers);
};

class NnVulkanDevice : public NnDevice {
private:
    NnVulkanContext context;
    NnVulkanData data;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
public:
    NnVulkanDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnVulkanDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    void syncPointers() override;
};

class NnVulkanDeviceSegment : public NnDeviceSegment {
private:
    NnVulkanContext *context;
    NnVulkanData *data;
    NnSegmentConfig *segmentConfig;
    std::vector<NnUint> weightBufferIndex;
public:
    NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanData *data, NnSegmentConfig *segmentConfig);
    ~NnVulkanDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif