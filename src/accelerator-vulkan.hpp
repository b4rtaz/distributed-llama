#ifndef ACCELERATOR_VULKAN_HPP
#define ACCELERATOR_VULKAN_HPP

#include <vulkan/vulkan.hpp>
#include "accelerator.hpp"

struct BufferVulkan {
    size_t bufferSize;
    vk::Buffer buffer;
    vk::DeviceMemory deviceMemory;
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
    vk::CommandBuffer commandBuffer;
};

class AcceleratorVulkan : public Accelerator {
private:
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    uint32_t queueFamilyIndex;
    vk::CommandPool commandPool;

    std::vector<MatmulVulkan> matmuls;

public:
    AcceleratorVulkan();
    ~AcceleratorVulkan();
    unsigned int allocateMatmul(const FloatType inputFloatType, const FloatType weightsFloatType, const unsigned int n, const unsigned int d);
    void loadMatmulWeights(const unsigned int matmulIndex, const void* weights);
    void beginForwardMatmul(const unsigned int matmulIndex, const void* input);
    void endForwardMatmul(const unsigned int matmulIndex, float* output);
    void closeMatmul(const unsigned int matmulIndex);
};

#endif