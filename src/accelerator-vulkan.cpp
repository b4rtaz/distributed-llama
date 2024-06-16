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

#define MEMORY_TYPE_INDEX_NOT_FOUND ~0

uint32_t findMemoryTypeIndex(VulkanContext* context, vk::MemoryPropertyFlags expectedFlags) {
    vk::PhysicalDeviceMemoryProperties memoryProperties = context->physicalDevice.getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        auto flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((flags & expectedFlags) == expectedFlags) {
            return index;
        }
    }
    return MEMORY_TYPE_INDEX_NOT_FOUND;
}

void readShaderFile(const char* path, char** output, size_t* fileSize) {
    FILE* fd = fopen(path, "rb");
    if (fd == NULL)
        throw std::runtime_error("Cannot open shader file");
    *fileSize = seekToEnd(fd);
    rewind(fd);
    assert(*fileSize > 0);
    *output = new char[*fileSize];
    if (fread(*output, *fileSize, 1, fd) != 1)
        throw std::runtime_error("Cannot read shader file");
    assert(fclose(fd) == 0);
}

VulkanContext::VulkanContext() {
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
    deviceExtension.push_back("VK_KHR_16bit_storage");
    deviceExtension.push_back("VK_KHR_shader_float16_int8");

    vk::ApplicationInfo appInfo {"Distributed Llama", 1, nullptr, 0, VK_API_VERSION_1_2 };
    vk::InstanceCreateInfo instanceCreateInfo(createInstanceFlags, &appInfo, instanceLayers, instanceExtensions);
    instance = vk::createInstance(instanceCreateInfo);

    auto physicalDevices = instance.enumeratePhysicalDevices();
    physicalDevice = physicalDevices.front();
    if (physicalDevices.size() > 1)
        printf("WARN: found %lu Vulkan devices, chosen first\n", physicalDevices.size());

    vk::PhysicalDeviceProperties deviceProps = physicalDevice.getProperties();
    printf("🌋 device: %s\n", (char*)deviceProps.deviceName);
    printf("🌋 deviceApiVersion: %d.%d.%d\n", VK_VERSION_MAJOR(deviceProps.apiVersion), VK_VERSION_MINOR(deviceProps.apiVersion), VK_VERSION_PATCH(deviceProps.apiVersion));
    printf("🌋 maxComputeSharedMemory: %d kB\n", deviceProps.limits.maxComputeSharedMemorySize / 1024);

    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    for (unsigned int h = 0; h < memoryProperties.memoryHeapCount; h++) {
        if (memoryProperties.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            printf("🌋 heap[%u]: %lu MB\n", h, ((unsigned long)memoryProperties.memoryHeaps[h].size) / (1024 * 1024));
    }

    vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();

    VkPhysicalDeviceFeatures2 deviceFeatures2;
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = nullptr;
    deviceFeatures2.features = (VkPhysicalDeviceFeatures)deviceFeatures;

    VkPhysicalDeviceVulkan11Features vk11Features;
    vk11Features.pNext = nullptr;
    vk11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    deviceFeatures2.pNext = &vk11Features;

    VkPhysicalDeviceVulkan12Features vk12Features;
    vk12Features.pNext = nullptr;
    vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk11Features.pNext = &vk12Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures2);

    std::vector<vk::QueueFamilyProperties> queueFamilyProps = physicalDevice.getQueueFamilyProperties();
    auto propIt = std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties& Prop) {
        return Prop.queueFlags & vk::QueueFlagBits::eCompute;
    });
    queueFamilyIndex = std::distance(queueFamilyProps.begin(), propIt);

    const float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &queuePriority);

    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo);
    deviceCreateInfo.enabledExtensionCount = deviceExtension.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtension.data();
    deviceCreateInfo.setPNext(&deviceFeatures2);
    device = physicalDevice.createDevice(deviceCreateInfo);

    vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eTransient), queueFamilyIndex);
    commandPool = device.createCommandPool(commandPoolCreateInfo);

    queue = device.getQueue(queueFamilyIndex, 0);
}

VulkanContext::~VulkanContext() {
    device.destroyCommandPool(commandPool);
    device.destroy();
    instance.destroy();
}

std::pair<vk::Buffer, vk::DeviceMemory> createBuffer(VulkanContext* context, uint32_t memoryTypeIndex, const uint32_t bufferSize, vk::BufferUsageFlags usageFlags) {
	vk::PhysicalDeviceMemoryProperties memoryProperties = context->physicalDevice.getMemoryProperties();

	vk::BufferCreateInfo bufferCreateInfo {
		vk::BufferCreateFlags(), bufferSize, usageFlags,
		vk::SharingMode::eExclusive, 1, &context->queueFamilyIndex };
	vk::Buffer buffer = context->device.createBuffer(bufferCreateInfo);

	vk::MemoryRequirements memoryRequirements = context->device.getBufferMemoryRequirements(buffer);

	vk::MemoryAllocateInfo bufferMemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex);

	vk::DeviceMemory bufferMemory = context->device.allocateMemory(bufferMemoryAllocateInfo);

	context->device.bindBufferMemory(buffer, bufferMemory, 0);
	return std::make_pair(buffer, bufferMemory);
}

CopyBufferVulkan::CopyBufferVulkan(VulkanContext* context, const uint32_t bufferSize, vk::Buffer& deviceBuffer, CopyBufferVulkanDirection direction) {
    this->context = context;
    this->bufferSize = bufferSize;
    this->deviceBuffer = deviceBuffer;
    this->direction = direction;

    uint32_t memoryTypeIndex = findMemoryTypeIndex(context, vk::MemoryPropertyFlagBits::eHostVisible);
    if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
        throw std::runtime_error("Cannot find host visible memory type");

	auto b = createBuffer(context, memoryTypeIndex, bufferSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
	hostBuffer = b.first;
	hostMemory = b.second;

    vk::CommandBufferAllocateInfo allocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
	const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(allocInfo);
	commandBuffer = cmdBuffers.front();
	commandBuffer.begin({ vk::CommandBufferUsageFlags{} });

	VkBufferCopy copyRegion = { 0 };
	copyRegion.size = bufferSize;
    if (direction == FROM_HOST_TO_DEVICE) {
	    vkCmdCopyBuffer(commandBuffer, hostBuffer, deviceBuffer, 1, &copyRegion);
    } else if (direction == FROM_DEVICE_TO_HOST) {
        vkCmdCopyBuffer(commandBuffer, deviceBuffer, hostBuffer, 1, &copyRegion);
    }
	commandBuffer.end();

    fence = context->device.createFence(vk::FenceCreateInfo());
}

CopyBufferVulkan::~CopyBufferVulkan() {
	context->device.destroyFence(fence);
	context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);

	context->device.freeMemory(hostMemory);
	context->device.destroyBuffer(hostBuffer);
}

void CopyBufferVulkan::copy(void* data) {
    if (direction == FROM_HOST_TO_DEVICE) {
        void* local = context->device.mapMemory(hostMemory, 0, bufferSize);
        memcpy(local, data, bufferSize);
        context->device.unmapMemory(hostMemory);
    }

    context->device.resetFences({ fence });

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
	context->queue.submit({ submitInfo }, fence);
	auto result = context->device.waitForFences({ fence }, true, uint64_t(-1));

    if (direction == FROM_DEVICE_TO_HOST) {
        void* local = context->device.mapMemory(hostMemory, 0, bufferSize);
        memcpy(data, local, bufferSize);
        context->device.unmapMemory(hostMemory);
    }
}

BufferVulkan::BufferVulkan(VulkanContext* context, const uint32_t bufferSize, vk::BufferUsageFlags usageFlags, bool fastCopy, CopyBufferVulkanDirection direction) {
    this->context = context;
    this->bufferSize = bufferSize;

    uint32_t memoryTypeIndex = findMemoryTypeIndex(context, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (memoryTypeIndex != MEMORY_TYPE_INDEX_NOT_FOUND) {
        isHostVisible = true;
    } else {
        isHostVisible = false;
        memoryTypeIndex = findMemoryTypeIndex(context, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }
    if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
        throw std::runtime_error("Cannot find host visible memory type");

    auto b = createBuffer(context, memoryTypeIndex, bufferSize, usageFlags | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    deviceBuffer = b.first;
    deviceMemory = b.second;

    if (!isHostVisible && fastCopy)
        copy = new CopyBufferVulkan(context, bufferSize, deviceBuffer, direction);
    else
        copy = NULL;
}

void BufferVulkan::write(const void* data) {
    if (isHostVisible) {
        void* local = context->device.mapMemory(deviceMemory, 0, bufferSize);
        memcpy(local, data, bufferSize);
        context->device.unmapMemory(deviceMemory);
    } else if (copy != NULL) {
        assert(copy->direction == FROM_HOST_TO_DEVICE);
        copy->copy((void*)data);
    } else {
        CopyBufferVulkan cpy = CopyBufferVulkan(context, bufferSize, deviceBuffer, FROM_HOST_TO_DEVICE);
        cpy.copy((void*)data);
    }
}

void BufferVulkan::read(void* data) {
    if (isHostVisible) {
        void* local = context->device.mapMemory(deviceMemory, 0, bufferSize);
        memcpy(data, local, bufferSize);
        context->device.unmapMemory(deviceMemory);
    } else if (copy != NULL) {
        assert(copy->direction == FROM_DEVICE_TO_HOST);
        copy->copy(data);
    } else {
        throw std::runtime_error("Not implemented");
    }
}

void BufferVulkan::destroy() {
    this->context->device.freeMemory(deviceMemory);
    this->context->device.destroyBuffer(deviceBuffer);
    if (copy != NULL)
        delete copy;
}

AcceleratorVulkan::AcceleratorVulkan() : context(VulkanContext()) {
}

AcceleratorVulkan::~AcceleratorVulkan() {
    for (int m = 0; m < matmuls.size(); m++) {
        MatmulVulkan* mm = &matmuls[m];
        mm->inputBuffer.destroy();
        mm->weightsBuffer.destroy();
        mm->outputBuffer.destroy();
        mm->metadataBuffer.destroy();

        context.device.destroyDescriptorSetLayout(mm->descriptorSetLayout);
		context.device.destroyPipelineLayout(mm->pipelineLayout);
		context.device.destroyPipelineCache(mm->pipelineCache);
		context.device.destroyShaderModule(mm->shaderModule);
		context.device.destroyPipeline(mm->computePipeline);
		context.device.destroyDescriptorPool(mm->descriptorPool);
		context.device.destroyFence(mm->fence);

        delete mm->shaderData;
    }
}

unsigned int AcceleratorVulkan::allocateMatmul(const FloatType weightsFloatType, const FloatType inputFloatType, const unsigned int n, const unsigned int d) {
    assert(n % 32 == 0);

    const uint32_t inputBufferSize = getBatchBytes(inputFloatType, n, 1);
    const uint32_t weightsBufferSize = getBatchBytes(weightsFloatType, n, d);
    const uint32_t outputBufferSize = getBatchBytes(F32, d, 1);
    const uint32_t metadataBufferSize = sizeof(uint32_t);

    MatmulVulkan mm = {
        .inputBuffer = BufferVulkan(&context, inputBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, true, FROM_HOST_TO_DEVICE),
        .weightsBuffer = BufferVulkan(&context, weightsBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, false, FROM_HOST_TO_DEVICE),
        .outputBuffer = BufferVulkan(&context, outputBufferSize, vk::BufferUsageFlagBits::eStorageBuffer, true, FROM_DEVICE_TO_HOST),
        .metadataBuffer = BufferVulkan(&context, metadataBufferSize, vk::BufferUsageFlagBits::eUniformBuffer, false, FROM_HOST_TO_DEVICE)
    };

    size_t shaderSize;
    if (inputFloatType == F32 && weightsFloatType == F32) {
        readShaderFile("src/vulkan/matmul-f32-f32.spv", &mm.shaderData, &shaderSize);
    } else if (inputFloatType == Q80 && weightsFloatType == Q40) {
        readShaderFile("src/vulkan/matmul-q40-q80.spv", &mm.shaderData, &shaderSize);
    } else if (inputFloatType == F32 && weightsFloatType == Q40) {
        readShaderFile("src/vulkan/matmul-q40-f32.spv", &mm.shaderData, &shaderSize);
    } else {
        throw std::runtime_error("Unsupported float type: " + std::to_string(inputFloatType) + "/" + std::to_string(weightsFloatType));
    }

    vk::ShaderModuleCreateInfo ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shaderSize, (uint32_t*)mm.shaderData);
    mm.shaderModule = context.device.createShaderModule(ShaderModuleCreateInfo);

    const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBinding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };
    vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), descriptorSetLayoutBinding);
    mm.descriptorSetLayout = context.device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), mm.descriptorSetLayout);
    mm.pipelineLayout = context.device.createPipelineLayout(pipelineLayoutCreateInfo);
    mm.pipelineCache = context.device.createPipelineCache(vk::PipelineCacheCreateInfo());

    vk::PipelineShaderStageCreateInfo pipelineShaderCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, mm.shaderModule, "main");
    vk::ComputePipelineCreateInfo computePipelineCreateInfo(vk::PipelineCreateFlags(), pipelineShaderCreateInfo, mm.pipelineLayout);
    mm.computePipeline = context.device.createComputePipeline(mm.pipelineCache, computePipelineCreateInfo).value;

    vk::DescriptorPoolSize descriptorPoolSizes[2];
    descriptorPoolSizes[0] = { vk::DescriptorType::eStorageBuffer, 3 };
    descriptorPoolSizes[1] = { vk::DescriptorType::eUniformBuffer, 1 };

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), 2, descriptorPoolSizes);
    mm.descriptorPool = context.device.createDescriptorPool(descriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(mm.descriptorPool, 1, &mm.descriptorSetLayout);
    const std::vector<vk::DescriptorSet> descriptorSets = context.device.allocateDescriptorSets(descriptorSetAllocInfo);
    vk::DescriptorSet descriptorSet = descriptorSets.front();
    vk::DescriptorBufferInfo inputBufferInfo(mm.inputBuffer.deviceBuffer, 0, inputBufferSize);
    vk::DescriptorBufferInfo weightsBufferInfo(mm.weightsBuffer.deviceBuffer, 0, weightsBufferSize);
    vk::DescriptorBufferInfo outputBufferInfo(mm.outputBuffer.deviceBuffer, 0, outputBufferSize);
    vk::DescriptorBufferInfo metadataBufferInfo(mm.metadataBuffer.deviceBuffer, 0, metadataBufferSize);

    const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
        {descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputBufferInfo},
        {descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &weightsBufferInfo},
        {descriptorSet, 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outputBufferInfo},
        {descriptorSet, 3, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &metadataBufferInfo},
    };
    context.device.updateDescriptorSets(writeDescriptorSets, {});

    vk::CommandBufferAllocateInfo commandBufferAllocInfo(context.commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context.device.allocateCommandBuffers(commandBufferAllocInfo);
    mm.commandBuffer = cmdBuffers.front();

    mm.commandBuffer.begin({ vk::CommandBufferUsageFlags{} });

    mm.commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, mm.computePipeline);
    mm.commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, mm.pipelineLayout, 0, { descriptorSet }, {});
    mm.commandBuffer.dispatch(1, d, 1);
    mm.commandBuffer.end();

    mm.fence = context.device.createFence(vk::FenceCreateInfo());

    uint32_t metadata = (uint32_t)n;
    mm.metadataBuffer.write((const void*)&metadata);

    matmuls.push_back(mm);
    return matmuls.size() - 1;
}

void AcceleratorVulkan::loadMatmulWeights(const unsigned int matmulIndex, const void* weights) {
    auto mm = &matmuls[matmulIndex];
    mm->weightsBuffer.write(weights);
}

void AcceleratorVulkan::beginForwardMatmul(const unsigned int matmulIndex, const void* input) {
    auto mm = &matmuls[matmulIndex];
    mm->inputBuffer.write(input);

    context.device.resetFences({ mm->fence });
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &mm->commandBuffer);
    context.queue.submit({ submitInfo }, mm->fence);
}

void AcceleratorVulkan::endForwardMatmul(const unsigned int matmulIndex, float* output) {
    auto mm = &matmuls[matmulIndex];
    context.device.waitForFences({ mm->fence }, true, uint64_t(-1));

    mm->outputBuffer.read((void*)output);
}
