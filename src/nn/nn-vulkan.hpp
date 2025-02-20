#ifndef NN_VULKAN_HPP
#define NN_VULKAN_HPP

#include <vulkan/vulkan.hpp>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

typedef struct {
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    uint32_t queueFamilyIndex;
    vk::CommandPool commandPool;
    vk::Queue queue;
} NnVulkanContext;

class NnVulkanDevice : public NnDevice {
private:
    NnVulkanContext context;
public:
    NnVulkanDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnVulkanDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
    void syncPointers() override;
};

class NnVulkanDeviceSegment : public NnDeviceSegment {
private:
    NnVulkanContext context;
public:
    NnVulkanDeviceSegment(NnVulkanContext *context);
    ~NnVulkanDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif