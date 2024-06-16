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

class BufferVulkan {
public:
    VulkanContext* context;
    size_t bufferSize;
    vk::Buffer buffer;
    vk::DeviceMemory deviceMemory;

    BufferVulkan(VulkanContext* context, const uint32_t bufferSize, vk::BufferUsageFlags usageFlags);
    void load(const void* data);
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
    VulkanContext* context;
    std::vector<MatmulVulkan> matmuls;

public:
    AcceleratorVulkan(VulkanContext* context);
    ~AcceleratorVulkan();
    unsigned int allocateMatmul(const FloatType weightsFloatType, const FloatType inputFloatType, const unsigned int n, const unsigned int d);
    void loadMatmulWeights(const unsigned int matmulIndex, const void* weights);
    void beginForwardMatmul(const unsigned int matmulIndex, const void* input);
    void endForwardMatmul(const unsigned int matmulIndex, float* output);
};

#endif