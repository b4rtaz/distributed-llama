#ifndef ACCELERATOR_VULKAN_HPP
#define ACCELERATOR_VULKAN_HPP

#include <vulkan/vulkan.hpp>
#include "accelerator.hpp"

class VulkanContext {
public:
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    uint32_t queueFamilyIndex;
    vk::CommandPool commandPool;
    vk::Queue queue;

    VulkanContext();
    ~VulkanContext();
};

enum CopyBufferVulkanDirection {
    FROM_HOST_TO_DEVICE,
    FROM_DEVICE_TO_HOST
};

class CopyBufferVulkan {
public:
    VulkanContext* context;
    uint32_t bufferSize;
    vk::Buffer deviceBuffer;
    vk::Buffer hostBuffer;
    vk::DeviceMemory hostMemory;
    vk::Fence fence;
    vk::CommandBuffer commandBuffer;
    bool direction;
    void *hostPointer;
    CopyBufferVulkan(VulkanContext* context, const uint32_t bufferSize, vk::Buffer& deviceBuffer, CopyBufferVulkanDirection direction);
    ~CopyBufferVulkan();
    void copy(void* data);
};

class BufferVulkan {
public:
    VulkanContext* context;
    size_t bufferSize;
    vk::Buffer deviceBuffer;
    vk::DeviceMemory deviceMemory;
    bool isHostVisible;
    CopyBufferVulkan* copy;
    void *hostPointer;

    BufferVulkan(VulkanContext* context, const uint32_t bufferSize, vk::BufferUsageFlags usageFlags, bool fastCopy, CopyBufferVulkanDirection direction);
    void write(const void* data);
    void read(void* data);
    void destroy();
};

struct MatmulVulkan {
    BufferVulkan inputBuffer;
    BufferVulkan weightsBuffer;
    BufferVulkan outputBuffer;
    BufferVulkan metadataBuffer;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    vk::Pipeline computePipeline;
    vk::PipelineCache pipelineCache;
    vk::PipelineLayout pipelineLayout;
    vk::ShaderModule shaderModule;
    char* shaderData;
    vk::CommandBuffer commandBuffer;
    vk::Fence fence;
};

class AcceleratorVulkan : public Accelerator {
private:
    VulkanContext context;
    std::vector<MatmulVulkan> matmuls;

public:
    AcceleratorVulkan();
    ~AcceleratorVulkan();
    unsigned int allocateMatmul(const FloatType weightsFloatType, const FloatType inputFloatType, const unsigned int n, const unsigned int d);
    void loadMatmulWeights(const unsigned int matmulIndex, const void* weights);
    void beginForwardMatmul(const unsigned int matmulIndex, const void* input);
    void endForwardMatmul(const unsigned int matmulIndex, float* output);
};

#endif