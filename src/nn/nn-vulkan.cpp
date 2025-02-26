#include "nn-vulkan.hpp"

#if DEBUG_VULKAN_TRACE
    #define VULKAN_TRACE(...) printf("VULKAN_TRACE: "); printf(__VA_ARGS__); printf("\n");
#else
    #define VULKAN_TRACE(...)
#endif

static bool hasPortabilityExtension() {
#ifdef __APPLE__
    const std::vector<vk::ExtensionProperties> extensionProperties = vk::enumerateInstanceExtensionProperties();
    for (const auto& extension : extensionProperties) {
        if (std::strcmp(extension.extensionName, "VK_KHR_portability_enumeration") == 0)
            return true;
    }
#endif
    return false;
}

static bool hasValidationLayer() {
    const std::vector<vk::LayerProperties> layerProperties = vk::enumerateInstanceLayerProperties();
    for (const auto& layer : layerProperties) {
        if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
            return true;
    }
    return false;
}

#define MEMORY_TYPE_INDEX_NOT_FOUND ~0

static uint32_t findMemoryTypeIndex(const vk::PhysicalDevice *physicalDevice, vk::MemoryPropertyFlags expectedFlags) {
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice->getMemoryProperties();
    for (uint32_t index = 0; index < memoryProperties.memoryTypeCount; index++) {
        auto flags = memoryProperties.memoryTypes[index].propertyFlags;
        if ((flags & expectedFlags) == expectedFlags) {
            return index;
        }
    }
    return MEMORY_TYPE_INDEX_NOT_FOUND;
}

static std::pair<vk::Buffer, vk::DeviceMemory> createBuffer(const NnVulkanContext *context, const uint32_t memoryTypeIndex, const vk::DeviceSize bufferSize, const vk::BufferUsageFlags usageFlags) {
    vk::BufferCreateInfo bufferCreateInfo {
        vk::BufferCreateFlags(),
        bufferSize,
        usageFlags,
        vk::SharingMode::eExclusive,
        1,
        &context->queueFamilyIndex
    };
    vk::Buffer buffer = context->device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = context->device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo bufferMemoryAllocateInfo(memoryRequirements.size, memoryTypeIndex);
    vk::DeviceMemory bufferMemory = context->device.allocateMemory(bufferMemoryAllocateInfo);

    context->device.bindBufferMemory(buffer, bufferMemory, 0);
    return std::make_pair(buffer, bufferMemory);
}

NnVulkanStagingCopy::NnVulkanStagingCopy(const NnVulkanContext *context, vk::Buffer& deviceBuffer, const vk::DeviceSize bufferSize, const NnStagingVulkanCopyDirection direction) {
    this->direction = direction;
    this->deviceBuffer = deviceBuffer;
    this->context = context;
    this->bufferSize = bufferSize;

    uint32_t memoryTypeIndex = findMemoryTypeIndex(&context->physicalDevice, vk::MemoryPropertyFlagBits::eHostVisible);
    if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
        throw std::runtime_error("Cannot find host visible memory type");
    auto b = createBuffer(context, memoryTypeIndex, bufferSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    hostBuffer = b.first;
    hostMemory = b.second;
    hostPointer = context->device.mapMemory(hostMemory, 0, bufferSize);
}

NnVulkanStagingCopy::~NnVulkanStagingCopy() {
    context->device.unmapMemory(hostMemory);
    context->device.freeMemory(hostMemory);
    context->device.destroyBuffer(hostBuffer);
}

void NnVulkanStagingCopy::copy(NnByte *data) {
    switch (direction) {
    case COPY_TO_DEVICE:
        std::memcpy(hostPointer, data, bufferSize);
        break;
    case COPY_FROM_DEVICE:
        std::memcpy(data, hostPointer, bufferSize);
        break;
    }
}

void NnVulkanStagingCopy::addCopyCommand(vk::CommandBuffer& commandBuffer) {
    VkBufferCopy copyRegion = { 0 };
	copyRegion.size = bufferSize;
    switch (direction) {
    case COPY_TO_DEVICE:
        vkCmdCopyBuffer(commandBuffer, hostBuffer, deviceBuffer, 1, &copyRegion);
        break;
    case COPY_FROM_DEVICE:
        vkCmdCopyBuffer(commandBuffer, deviceBuffer, hostBuffer, 1, &copyRegion);
        break;
    }
}

NnVulkanBuffer::NnVulkanBuffer(NnVulkanContext *context, const vk::DeviceSize bufferSize, vk::BufferUsageFlags usageFlags, bool fastAccess) {
    this->context = context;
    this->bufferSize = bufferSize;
    this->usageFlags = usageFlags;
    this->hostPointer = nullptr;

    uint32_t memoryTypeIndex = MEMORY_TYPE_INDEX_NOT_FOUND;
    if (fastAccess) {
        memoryTypeIndex = findMemoryTypeIndex(&context->physicalDevice, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eDeviceLocal);
        if (memoryTypeIndex != MEMORY_TYPE_INDEX_NOT_FOUND)
            isHostVisible = true;
    }
    if (!isHostVisible) {
        memoryTypeIndex = findMemoryTypeIndex(&context->physicalDevice, vk::MemoryPropertyFlagBits::eDeviceLocal);
        if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
            throw std::runtime_error("Cannot find host visible memory type");
    }

    auto b = createBuffer(context, memoryTypeIndex, bufferSize, usageFlags | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    deviceBuffer = b.first;
    deviceMemory = b.second;
    if (isHostVisible)
        hostPointer = context->device.mapMemory(deviceMemory, 0, bufferSize);
    VULKAN_TRACE("Created buffer of size %lld (fastAccess=%d)", bufferSize, fastAccess);
}

NnVulkanBuffer::~NnVulkanBuffer() {
    if (hostPointer != nullptr)
        context->device.unmapMemory(deviceMemory);
    context->device.freeMemory(deviceMemory);
    context->device.destroyBuffer(deviceBuffer);
    VULKAN_TRACE("Destroyed buffer of size %lld", bufferSize);
}

void NnVulkanBuffer::write(const NnByte *data) {
    if (isHostVisible && hostPointer != nullptr) {
        std::memcpy(hostPointer, data, bufferSize);
    } else {
        NnVulkanStagingCopy copy(context, deviceBuffer, bufferSize, COPY_TO_DEVICE);
        copy.copy((NnByte *)data);

        vk::CommandBufferAllocateInfo allocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
        const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(allocInfo);
        vk::CommandBuffer commandBuffer = cmdBuffers.front();
        commandBuffer.begin({ vk::CommandBufferUsageFlags{} });
        copy.addCopyCommand(commandBuffer);
        commandBuffer.end();

        vk::Fence fence = context->device.createFence(vk::FenceCreateInfo());
        vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
        context->queue.submit({ submitInfo }, fence);
        assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);

        context->device.destroyFence(fence);
        context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
    }
    VULKAN_TRACE("Wrote %lld bytes to buffer", bufferSize);
}

void NnVulkanBuffer::read(NnByte *data) {
    // TODO: this function should be deleted
    assert(isHostVisible && hostPointer != nullptr);
    std::memcpy(data, hostPointer, bufferSize);
    VULKAN_TRACE("Read %lld bytes from buffer", bufferSize);
}

NnVulkanData::NnVulkanData(NnVulkanContext *context, NnNetConfig *netConfig, NnNodeConfig *nodeConfig)
    : pipes(netConfig->nPipes), buffers(nodeConfig->nBuffers), internalBuffers()
{
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;

    for (NnUint i = 0; i < netConfig->nPipes; i++)
        pipes[i].reset(new NnVulkanBuffer(context, netConfig->pipes[i].size.nBytes, vk::BufferUsageFlagBits::eStorageBuffer, true));
    for (NnUint i = 0; i < nodeConfig->nBuffers; i++)
        buffers[i].reset(new NnVulkanBuffer(context, nodeConfig->buffers[i].size.nBytes, vk::BufferUsageFlagBits::eStorageBuffer, false));
}

NnVulkanData::~NnVulkanData() {
    pipes.clear();
    buffers.clear();
    internalBuffers.clear();
}

NnSize2D NnVulkanData::resolvePointerSize(NnPointerConfig *config) {
    if (config->pointerType == PNTR_BUFFER)
        return nodeConfig->buffers[config->pointerIndex].size;
    if (config->pointerType == PNTR_PIPE)
        return netConfig->pipes[config->pointerIndex].size;
    throw std::invalid_argument("Unsupported pointer config");
}

NnVulkanBuffer *NnVulkanData::resolveBuffer(NnPointerConfig *config) {
    if (config->pointerType == PNTR_BUFFER)
        return buffers[config->pointerIndex].get();
    if (config->pointerType == PNTR_PIPE)
        return pipes[config->pointerIndex].get();
    throw std::invalid_argument("Unsupported pointer config");
}

NnVulkanDevice::NnVulkanDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;
    this->netExecution = netExecution;

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
    context.instance = vk::createInstance(instanceCreateInfo);

    auto physicalDevices = context.instance.enumeratePhysicalDevices();
    context.physicalDevice = physicalDevices.front();

    vk::PhysicalDeviceProperties deviceProps = context.physicalDevice.getProperties();
    printf("🌋 Device: %s\n", (char*)deviceProps.deviceName);
    printf("🌋 DeviceApiVersion: %d.%d.%d\n", VK_VERSION_MAJOR(deviceProps.apiVersion), VK_VERSION_MINOR(deviceProps.apiVersion), VK_VERSION_PATCH(deviceProps.apiVersion));
    printf("🌋 MaxComputeSharedMemory: %d kB\n", deviceProps.limits.maxComputeSharedMemorySize / 1024);

    vk::PhysicalDeviceMemoryProperties memoryProperties = context.physicalDevice.getMemoryProperties();
    for (unsigned int h = 0; h < memoryProperties.memoryHeapCount; h++) {
        if (memoryProperties.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            printf("🌋 Heap[%u]: %lu MB\n", h, ((unsigned long)memoryProperties.memoryHeaps[h].size) / (1024 * 1024));
    }

    vk::PhysicalDeviceFeatures deviceFeatures = context.physicalDevice.getFeatures();

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
    vkGetPhysicalDeviceFeatures2(context.physicalDevice, &deviceFeatures2);

    std::vector<vk::QueueFamilyProperties> queueFamilyProps = context.physicalDevice.getQueueFamilyProperties();
    auto propIt = std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties& Prop) {
        return Prop.queueFlags & vk::QueueFlagBits::eCompute;
    });
    context.queueFamilyIndex = std::distance(queueFamilyProps.begin(), propIt);

    const float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), context.queueFamilyIndex, 1, &queuePriority);

    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), deviceQueueCreateInfo);
    deviceCreateInfo.enabledExtensionCount = deviceExtension.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtension.data();
    deviceCreateInfo.setPNext(&deviceFeatures2);
    context.device = context.physicalDevice.createDevice(deviceCreateInfo);

    vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eTransient), context.queueFamilyIndex);
    context.commandPool = context.device.createCommandPool(commandPoolCreateInfo);
    context.queue = context.device.getQueue(context.queueFamilyIndex, 0);

    data = new NnVulkanData(&context, netConfig, nodeConfig);
}

NnVulkanDevice::~NnVulkanDevice() {
    delete data;

    context.device.destroyCommandPool(context.commandPool);
    context.device.destroy();
    context.instance.destroy();
}

NnUint NnVulkanDevice::maxNThreads() {
    return 1;
}

NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    return new NnVulkanDeviceSegment(&context, data, segmentConfig, netExecution);
};

void NnVulkanDevice::syncPointers() {
}

static const char *getShaderFileName(const NnOpCode opCode, const NnOpQuantType quantType) {
    if (opCode == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return "inv-rms-f32-f32.spv";
    }
    if (opCode == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return "rms-norm-f32-f32-f32.spv";
    }
    return nullptr;
}

NnVulkanShader::NnVulkanShader(const char *fileName) {
    std::string path = std::string("./src/nn/vulkan/") + fileName;
    FILE* file = fopen(path.c_str(), "rb");
    if (!file)
        throw std::runtime_error("Failed to open shader file: " + path);
    constexpr size_t chunkSize = 4096;
    uint32_t chunk[chunkSize];
    size_t bytesRead;
    while ((bytesRead = fread(chunk, 1, chunkSize, file)) > 0) {
        code.insert(code.end(), chunk, chunk + bytesRead);
    }
    if (ferror(file)) {
        fclose(file);
        throw std::runtime_error("Failed to read shader file: " + path);
    }
    fclose(file);
}

static vk::DescriptorType toDescriptorType(NnVulkanBuffer *buffer) {
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eUniformBuffer)
        return vk::DescriptorType::eUniformBuffer;
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eStorageBuffer)
        return vk::DescriptorType::eStorageBuffer;
    throw std::invalid_argument("Unsupported buffer usage");
}

static void modifySpvSet(std::vector<uint32_t>& binary, uint32_t new_set) {
    if (binary.size() < 5)
        throw std::runtime_error("Invalid SPIR-V binary: too short");

    uint32_t magic = binary[0];
    if (magic != 0x07230203)
        throw std::runtime_error("Unsupported endianness or not a SPIR-V binary");

    size_t index = 5;
    while (index < binary.size()) {
        uint32_t firstWord = binary[index];
        uint16_t opcode = firstWord & 0xFFFF;
        uint16_t wordCount = firstWord >> 16;
        if (wordCount == 0) break;
        if (opcode == 71 && wordCount >= 4) {
            uint32_t decoration = binary[index + 2];
            if (decoration == 34)
                binary[index + 3] = new_set;
        }
        index += wordCount;
    }
}

NnVulkanDeviceSegment::NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanData *data, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution) :
    weightBufferIndex(segmentConfig->nOps, UINT32_MAX),
    configBufferIndex(segmentConfig->nOps, UINT32_MAX),
    shaderModules(segmentConfig->nOps),
    descriptorSets(segmentConfig->nOps),
    descriptorPools(segmentConfig->nOps),
    descriptorSetLayouts(segmentConfig->nOps),
    groupCountX(segmentConfig->nOps)
{
    this->context = context;
    this->data = data;
    this->segmentConfig = segmentConfig;
    this->netExecution = netExecution;

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (opConfig->weightSize.nBytes > 0) {
            NnVulkanBuffer *buffer = new NnVulkanBuffer(context, opConfig->weightSize.nBytes, vk::BufferUsageFlagBits::eStorageBuffer, false);
            data->internalBuffers.push_back(std::unique_ptr<NnVulkanBuffer>(buffer));
            weightBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
        if (opConfig->configSize > 0) {
            NnVulkanBuffer *buffer = new NnVulkanBuffer(context, opConfig->configSize, vk::BufferUsageFlagBits::eUniformBuffer, false);
            data->internalBuffers.push_back(std::unique_ptr<NnVulkanBuffer>(buffer));
            buffer->write(opConfig->config);
            configBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
    }

    std::vector<vk::PipelineShaderStageCreateInfo> shaderCreateInfos(segmentConfig->nOps);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize2D inputSize = data->resolvePointerSize(&opConfig->input);
        NnSize2D outputSize = data->resolvePointerSize(&opConfig->output);
        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType);
        const char *shaderFileName = getShaderFileName(opConfig->code, opQuant);
        assert(shaderFileName != nullptr);
        NnVulkanShader shader(shaderFileName);
        modifySpvSet(shader.code, opIndex);

        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), shader.code.size(), shader.code.data());
        vk::ShaderModule shaderModule = context->device.createShaderModule(shaderModuleCreateInfo);

        vk::PipelineShaderStageCreateInfo shaderCreateInfo(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, shaderModule, "main");

        std::vector<NnVulkanBuffer *> buffers;
        {
            // input
            buffers.push_back(data->resolveBuffer(&opConfig->input));
            // output
            buffers.push_back(data->resolveBuffer(&opConfig->output));
            // weight
            if (opConfig->weightSize.nBytes > 0) {
                assert(weightBufferIndex[opIndex] != UINT32_MAX);
                buffers.push_back(data->internalBuffers[weightBufferIndex[opIndex]].get());
            }
            // config
            if (opConfig->configSize > 0) {
                assert(configBufferIndex[opIndex] != UINT32_MAX);
                buffers.push_back(data->internalBuffers[configBufferIndex[opIndex]].get());
            }
        }

        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings(buffers.size());
        for (NnUint i = 0; i < buffers.size(); i++)
            descriptorSetLayoutBindings[i] = vk::DescriptorSetLayoutBinding(i, toDescriptorType(buffers[i]), 1, vk::ShaderStageFlagBits::eCompute);

        vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), descriptorSetLayoutBindings.size(), descriptorSetLayoutBindings.data());
        vk::DescriptorSetLayout descriptorSetLayout = context->device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

        NnUint nUniformBuffers = 0;
        NnUint nStorageBuffers = 0;
        for (NnUint i = 0; i < buffers.size(); i++) {
            vk::DescriptorType descriptorType = toDescriptorType(buffers[i]);
            if (descriptorType == vk::DescriptorType::eUniformBuffer)
                nUniformBuffers++;
            if (descriptorType == vk::DescriptorType::eStorageBuffer)
                nStorageBuffers++;
        }

        std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
        if (nStorageBuffers > 0)
            descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, nStorageBuffers));
        if (nUniformBuffers > 0)
            descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, nUniformBuffers));

        vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, descriptorPoolSizes.size(), descriptorPoolSizes.data());
        vk::DescriptorPool descriptorPool = context->device.createDescriptorPool(descriptorPoolCreateInfo);
        vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(descriptorPool, 1, &descriptorSetLayout);
        const std::vector<vk::DescriptorSet> allocatedDescriptorSets = context->device.allocateDescriptorSets(descriptorSetAllocInfo);
        assert(allocatedDescriptorSets.size() == 1);

        vk::DescriptorSet descriptorSet = allocatedDescriptorSets[0];
        std::vector<vk::DescriptorBufferInfo> bufferInfos(buffers.size());
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets(buffers.size());
        for (NnUint i = 0; i < buffers.size(); i++) {
            bufferInfos[i] = vk::DescriptorBufferInfo(buffers[i]->deviceBuffer, 0, buffers[i]->bufferSize);
            writeDescriptorSets[i] = vk::WriteDescriptorSet(descriptorSet, i, 0, 1, toDescriptorType(buffers[i]), nullptr, &bufferInfos[i], nullptr);
        }

        context->device.updateDescriptorSets(writeDescriptorSets, nullptr);

        shaderModules[opIndex] = shaderModule;
        shaderCreateInfos[opIndex] = shaderCreateInfo;
        descriptorSets[opIndex] = descriptorSet;
        descriptorPools[opIndex] = descriptorPool;
        descriptorSetLayouts[opIndex] = descriptorSetLayout;
        groupCountX[opIndex] = inputSize.x / 32;
        VULKAN_TRACE("Shader %d groupCountX=%d", opIndex, groupCountX[opIndex]);
    }

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), descriptorSetLayouts.size(), descriptorSetLayouts.data());
    pipelineLayout = context->device.createPipelineLayout(pipelineLayoutCreateInfo);
    pipelineCache = context->device.createPipelineCache(vk::PipelineCacheCreateInfo());

    std::vector<vk::ComputePipelineCreateInfo> pipelineInfos(segmentConfig->nOps);
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        vk::ComputePipelineCreateInfo pipelineInfo(vk::PipelineCreateFlags(), shaderCreateInfos[opIndex], pipelineLayout);
        pipelineInfos[opIndex] = pipelineInfo;
    }

    pipelines = context->device.createComputePipelines(pipelineCache, pipelineInfos).value;
    fence = context->device.createFence(vk::FenceCreateInfo());
}

NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++)
        context->device.destroyPipeline(pipelines[opIndex]);
    context->device.destroyFence(fence);
    context->device.destroyPipelineLayout(pipelineLayout);
    context->device.destroyPipelineCache(pipelineCache);
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        context->device.destroyDescriptorPool(descriptorPools[opIndex]);
        context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]);
        context->device.destroyShaderModule(shaderModules[opIndex]);
    }
}

void NnVulkanDeviceSegment::loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) {
    assert(segmentConfig->nOps > opIndex);
    assert(segmentConfig->ops[opIndex].weightSize.nBytes == nBytes);
    NnUint dataBufferIndex = weightBufferIndex[opIndex];
    data->internalBuffers[dataBufferIndex]->write(weight);
}

void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    assert(threadIndex == 0);
    if (opIndex != 0) {
        // TODO: this is a design problem, executor tries to forward all ops in a segment
        return;
    }

    {
        // TODO
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->input.pointerType == PNTR_PIPE)
                data->pipes[opConfig->input.pointerIndex]->write(netExecution->pipes[opConfig->input.pointerIndex]);
        }
    }

    vk::CommandBufferAllocateInfo commandBufferAllocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(commandBufferAllocInfo);
    vk::CommandBuffer commandBuffer = cmdBuffers.front();

    vk::CommandBufferBeginInfo commandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer.begin(commandBufferBeginInfo);
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[opIndex]);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, opIndex, { descriptorSets[opIndex] }, {});
        commandBuffer.dispatch(groupCountX[opIndex], 1, 1);
    }
    commandBuffer.end();

    context->device.resetFences({ fence });
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    context->queue.submit({ submitInfo }, fence);
    assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);

    VULKAN_TRACE("Forwarded");

    {
        // TODO
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->output.pointerType == PNTR_PIPE)
                data->pipes[opConfig->output.pointerIndex]->read(netExecution->pipes[opConfig->output.pointerIndex]);
        }
    }
}
