#include <cstdio>
#include "utils.hpp"
#include "accelerator-vulkan.hpp"

bool hasPortabilityExtension() {
#ifdef __APPLE__
    const std::vector<vk::ExtensionProperties> extensionProperties = vk::enumerateInstanceExtensionProperties();
    for (const auto& extension : extensionProperties) {
        if (strcmp(extension.extensionName, "VK_KHR_portability_enumeration") == 0)
            return true;
    }
#endif
    return false;
}

bool hasValidationLayer() {
    const std::vector<vk::LayerProperties> layerProperties = vk::enumerateInstanceLayerProperties();
    for (const auto& layer : layerProperties) {
        if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
            return true;
    }
    return false;
}

char* readShaderFile(const char* path, size_t* fileSize) {
    FILE* fd = fopen(path, "rb");
    if (fd == NULL)
        throw new std::runtime_error("Cannot open shader file");
    *fileSize = seekToEnd(fd);
    rewind(fd);
    char* buffer = new char[*fileSize];
    if (fread(buffer, *fileSize, 1, fd) != 1)
        throw std::runtime_error("Cannot read shader file");
    return buffer;
}

BufferVulkan createBuffer(
    vk::PhysicalDevice& physicalDevice,
    vk::Device& device,
    vk::MemoryPropertyFlags memFlagBits,
    const uint32_t& computeQueueFamilyIndex,
    const uint32_t bufferSize,
    vk::BufferUsageFlags usageFlags
) {
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

    uint32_t memoryTypeIndex = uint32_t(~0);
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        auto flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((flags & memFlagBits) == memFlagBits) {
            memoryTypeIndex = index;
            break;
        }
    }
    if (memoryTypeIndex == uint32_t(~0))
        throw std::runtime_error("Cannot find required Vulkan memory type");

    vk::BufferCreateInfo bufferCreateInfo {
        vk::BufferCreateFlags(), bufferSize, usageFlags,
        vk::SharingMode::eExclusive, 1, &computeQueueFamilyIndex };
    vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo bufferMemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex);
    vk::DeviceMemory bufferMemory = device.allocateMemory(bufferMemoryAllocateInfo);
    device.bindBufferMemory(buffer, bufferMemory, 0);
    return { bufferSize, buffer, bufferMemory };
}

void destroyBuffer(vk::Device& device, BufferVulkan& bv) {
    device.freeMemory(bv.deviceMemory);
    device.destroyBuffer(bv.buffer);
}

AcceleratorVulkan::AcceleratorVulkan() {
    vk::InstanceCreateFlags createInstanceFlags(0);
    std::vector<const char*> instanceLayers = {};
    std::vector<const char*> instanceExtensions = {};
    std::vector<const char*> deviceExtension = {};

    if (hasValidationLayer()) {
        instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
    }
    if (hasPortabilityExtension()) {
        createInstanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        instanceExtensions.push_back("VK_KHR_portability_enumeration");
        deviceExtension.push_back("VK_KHR_portability_subset");
    }

    vk::ApplicationInfo appInfo {"Distributed Llama", 1, nullptr, 0, VK_API_VERSION_1_2 };
    vk::InstanceCreateInfo instanceCreateInfo(createInstanceFlags, &appInfo, instanceLayers, instanceExtensions);
    instance = vk::createInstance(instanceCreateInfo);

    physicalDevice = instance.enumeratePhysicalDevices()
        //.back();
        .front();

    vk::PhysicalDeviceProperties deviceProps = physicalDevice.getProperties();
    printf("🌋 device: %s\n", (char*)deviceProps.deviceName);
    printf("🌋 deviceApiVersion: %d.%d.%d\n", VK_VERSION_MAJOR(deviceProps.apiVersion), VK_VERSION_MINOR(deviceProps.apiVersion), VK_VERSION_PATCH(deviceProps.apiVersion));

    std::vector<vk::QueueFamilyProperties> queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    auto propIt = std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties& Prop) {
        return Prop.queueFlags & vk::QueueFlagBits::eCompute;
    });
    queueFamilyIndex = std::distance(queueFamilyProps.begin(), propIt);
    printf("🌋 queueFamilyIndex: %d\n", queueFamilyIndex);

    const float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);

    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo);
    deviceCreateInfo.enabledExtensionCount = deviceExtension.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtension.data();
    device = physicalDevice.createDevice(deviceCreateInfo);

    vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eTransient), queueFamilyIndex);
    commandPool = device.createCommandPool(commandPoolCreateInfo);
}

AcceleratorVulkan::~AcceleratorVulkan() {
    for (int m = 0; m < matmuls.size(); m++) {
        MatmulVulkan* mm = &matmuls[m];
        destroyBuffer(device, mm->inputBuffer);
        destroyBuffer(device, mm->weightsBuffer);
        destroyBuffer(device, mm->outputBuffer);
        destroyBuffer(device, mm->metadataBuffer);

        device.destroyDescriptorSetLayout(mm->descriptorSetLayout);
		device.destroyPipelineLayout(mm->pipelineLayout);
		device.destroyPipelineCache(mm->pipelineCache);
		device.destroyShaderModule(mm->shaderModule);
		device.destroyPipeline(mm->computePipeline);
		device.destroyDescriptorPool(mm->descriptorPool);
    }

    device.destroyCommandPool(commandPool);
    device.destroy();
    instance.destroy();
}

unsigned int AcceleratorVulkan::allocateMatmul(const FloatType inputFloatType, const FloatType weightsFloatType, const unsigned int n, const unsigned int d) {
    const uint32_t inputBufferSize = getBatchBytes(inputFloatType, n, 1);
    const uint32_t weightsBufferSize = getBatchBytes(weightsFloatType, n, d);
    const uint32_t outputBufferSize = getBatchBytes(F32, d, 1);
    const uint32_t metadataBufferSize = sizeof(uint32_t);

    MatmulVulkan mm;
    mm.inputBuffer = createBuffer(physicalDevice, device, vk::MemoryPropertyFlagBits::eDeviceLocal,
        queueFamilyIndex, inputBufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    mm.weightsBuffer = createBuffer(physicalDevice, device, vk::MemoryPropertyFlagBits::eDeviceLocal,
        queueFamilyIndex, weightsBufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    mm.outputBuffer = createBuffer(physicalDevice, device, vk::MemoryPropertyFlagBits::eDeviceLocal,
        queueFamilyIndex, outputBufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    mm.metadataBuffer = createBuffer(physicalDevice, device, vk::MemoryPropertyFlagBits::eDeviceLocal,
        queueFamilyIndex, metadataBufferSize, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);

    size_t shaderSize;
    char* shaderData = readShaderFile("src/vulkan/matmul-q40-q80.spv", &shaderSize);
    vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shaderSize, (uint32_t*)shaderData);
    mm.shaderModule = device.createShaderModule(ShaderModuleCreateInfo);
    delete[] shaderData;

    const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };
    vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), descriptorSetLayoutBinding);
    mm.descriptorSetLayout = device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), mm.descriptorSetLayout);
    mm.pipelineLayout = device.createPipelineLayout(pipelineLayoutCreateInfo);
    mm.pipelineCache = device.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, mm.shaderModule, "main");
    vk::ComputePipelineCreateInfo computePipelineCreateInfo(vk::PipelineCreateFlags(), pipelineShaderCreateInfo, mm.pipelineLayout);
    mm.computePipeline = device.createComputePipeline(mm.pipelineCache, computePipelineCreateInfo).value;

    vk::DescriptorPoolSize descriptorPoolSizes[2];
    descriptorPoolSizes[0] = { vk::DescriptorType::eStorageBuffer, 3 };
    descriptorPoolSizes[1] = { vk::DescriptorType::eUniformBuffer, 1 };

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 2, descriptorPoolSizes);
    mm.descriptorPool = device.createDescriptorPool(descriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(mm.descriptorPool, 1, &mm.descriptorSetLayout);
    const std::vector<vk::DescriptorSet> descriptorSets = device.allocateDescriptorSets(descriptorSetAllocInfo);
    vk::DescriptorSet descriptorSet = descriptorSets.front();
    vk::DescriptorBufferInfo inputBufferInfo(mm.inputBuffer.buffer, 0, inputBufferSize);
    vk::DescriptorBufferInfo weightsBufferInfo(mm.weightsBuffer.buffer, 0, weightsBufferSize);
    vk::DescriptorBufferInfo outputBufferInfo(mm.outputBuffer.buffer, 0, outputBufferSize);
    vk::DescriptorBufferInfo metadataBufferInfo(mm.metadataBuffer.buffer, 0, metadataBufferSize);

    const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
        {descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputBufferInfo},
        {descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &weightsBufferInfo},
        {descriptorSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outputBufferInfo},
        {descriptorSet, 3, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &metadataBufferInfo},
    };
    device.updateDescriptorSets(writeDescriptorSets, {});

    vk::CommandBufferAllocateInfo commandBufferAllocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = device.allocateCommandBuffers(commandBufferAllocInfo);
    mm.commandBuffer = cmdBuffers.front();

    mm.commandBuffer.begin({ vk::CommandBufferUsageFlags{} });

    mm.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, mm.computePipeline);
    mm.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, mm.pipelineLayout, 0, { descriptorSet }, {});
    mm.commandBuffer.dispatch(1, d / 4, 1);
    mm.commandBuffer.end();

    matmuls.push_back(mm);
    return matmuls.size() - 1;
}

void AcceleratorVulkan::loadMatmulWeights(const unsigned int matmulIndex, const void* weights) {}
void AcceleratorVulkan::beginForwardMatmul(const unsigned int matmulIndex, const void* input) {}
void AcceleratorVulkan::endForwardMatmul(const unsigned int matmulIndex, float* output) {}
void AcceleratorVulkan::closeMatmul(const unsigned int matmulIndex) {}
