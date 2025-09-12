#include "nn-vulkan.hpp"

#define DEBUG_VULKAN_BUFFERS false
#define DEBUG_VULKAN_TRACE false

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

static void assertDeviceExtensionSupport(const vk::PhysicalDevice &physicalDevice, const std::vector<const char *> &requiredExtensions) {
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    for (const auto& ext : requiredExtensions) {
        bool found = false;
        for (const auto& extension : availableExtensions) {
            if (std::strcmp(extension.extensionName, ext) == 0) {
                found = true;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(std::string("Device extension ") + ext + " is not supported");
    }
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
    hostPointer = (NnByte *)context->device.mapMemory(hostMemory, 0, bufferSize);
}

NnVulkanStagingCopy::~NnVulkanStagingCopy() {
    context->device.unmapMemory(hostMemory);
    context->device.freeMemory(hostMemory);
    context->device.destroyBuffer(hostBuffer);
}

void NnVulkanStagingCopy::copy(NnByte *data, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    switch (direction) {
    case COPY_TO_DEVICE:
        std::memcpy(&hostPointer[offset], data, size);
        break;
    case COPY_FROM_DEVICE:
        std::memcpy(data, &hostPointer[offset], size);
        break;
    }
}

void NnVulkanStagingCopy::addCopyCommand(vk::CommandBuffer& commandBuffer, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    VkBufferCopy copyRegion;
	copyRegion.dstOffset = offset;
	copyRegion.srcOffset = offset;
	copyRegion.size = size;
    switch (direction) {
    case COPY_TO_DEVICE:
        vkCmdCopyBuffer(commandBuffer, hostBuffer, deviceBuffer, 1, &copyRegion);
        break;
    case COPY_FROM_DEVICE:
        vkCmdCopyBuffer(commandBuffer, deviceBuffer, hostBuffer, 1, &copyRegion);
        break;
    }
}

void NnVulkanStagingCopy::executeCopyCommand(const NnSize offset, const NnSize size) {
    vk::CommandBufferAllocateInfo allocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(allocInfo);
    vk::CommandBuffer commandBuffer = cmdBuffers.front();
    commandBuffer.begin({ vk::CommandBufferUsageFlags{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit } });
    addCopyCommand(commandBuffer, offset, size);
    commandBuffer.end();

    vk::Fence fence = context->device.createFence(vk::FenceCreateInfo());
    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    context->queue.submit({ submitInfo }, fence);
    assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);

    context->device.destroyFence(fence);
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
}

NnVulkanBuffer::NnVulkanBuffer(NnVulkanContext *context, const char *name, const NnSize bufferSize, const bool isSliceable, vk::BufferUsageFlags usageFlags, bool fastAccess) {
    this->context = context;
    this->name = name;
    this->bufferSize = bufferSize;
    this->isSliceable = isSliceable;
    this->usageFlags = usageFlags;
    this->hostPointer = nullptr;

    isHostVisible = false;

    VULKAN_TRACE("Creating buffer of size %zu (fastAccess=%d)", bufferSize, fastAccess);

    uint32_t memoryTypeIndex = MEMORY_TYPE_INDEX_NOT_FOUND;
    if (fastAccess) {
        memoryTypeIndex = findMemoryTypeIndex(
            &context->physicalDevice,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent |
            vk::MemoryPropertyFlagBits::eHostCached
        );
        if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
            memoryTypeIndex = findMemoryTypeIndex(
                &context->physicalDevice,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent
            );
        if (memoryTypeIndex != MEMORY_TYPE_INDEX_NOT_FOUND)
            isHostVisible = true;
    }
    if (!isHostVisible) {
        memoryTypeIndex = findMemoryTypeIndex(
            &context->physicalDevice,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );
        if (memoryTypeIndex == MEMORY_TYPE_INDEX_NOT_FOUND)
            throw std::runtime_error("Cannot find host visible memory type");
    }

    auto b = createBuffer(context, memoryTypeIndex, bufferSize, usageFlags | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst);
    deviceBuffer = b.first;
    deviceMemory = b.second;
    if (isHostVisible)
        hostPointer = (NnByte *)context->device.mapMemory(deviceMemory, 0, bufferSize);
    VULKAN_TRACE("Created buffer of size %zu (fastAccess=%d)", bufferSize, fastAccess);
}

NnVulkanBuffer::~NnVulkanBuffer() {
    if (hostPointer != nullptr)
        context->device.unmapMemory(deviceMemory);
    context->device.freeMemory(deviceMemory);
    context->device.destroyBuffer(deviceBuffer);
    VULKAN_TRACE("Destroyed buffer of size %zu", bufferSize);
}

void NnVulkanBuffer::write(const NnByte *data) {
    write(data, 0u, bufferSize);
}

void NnVulkanBuffer::write(const NnByte *data, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    if (isHostVisible && hostPointer != nullptr) {
        std::memcpy(&hostPointer[offset], data, size);
        context->device.flushMappedMemoryRanges({ { deviceMemory, offset, (vk::DeviceSize)size } });
        VULKAN_TRACE("Wrote %zu bytes to host visible buffer", size);
    } else {
        NnVulkanStagingCopy copy(context, deviceBuffer, bufferSize, COPY_TO_DEVICE);
        copy.copy((NnByte *)data, offset, size);
        copy.executeCopyCommand(offset, size);
        VULKAN_TRACE("Wrote %zu bytes to buffer", size);
    }
}

void NnVulkanBuffer::read(NnByte *data) {
    read(data, 0u, bufferSize);
}

void NnVulkanBuffer::read(NnByte *data, const NnSize offset, const NnSize size) {
    assert(offset + size <= bufferSize);

    if (isHostVisible && hostPointer != nullptr) {
        context->device.invalidateMappedMemoryRanges({ {deviceMemory, 0, (vk::DeviceSize)size} });
        std::memcpy(data, hostPointer, size);

        VULKAN_TRACE("Read %zu bytes from host visible buffer", size);
    } else {
        NnVulkanStagingCopy copy(context, deviceBuffer, bufferSize, COPY_FROM_DEVICE);
        copy.executeCopyCommand(offset, size);
        copy.copy(data, offset, size);

        VULKAN_TRACE("Read %zu bytes from buffer", size);
    }
}

NnSize NnVulkanBuffer::calcSliceSize(const NnSize nominator, const NnSize denominator) {
    if (!isSliceable)
        return bufferSize;

    assert(bufferSize % denominator == 0);

    NnSize size = (bufferSize / denominator) * nominator;
    if (context->nonCoherentAtomSize != 0) {
        // TODO: this alignment is not needed for coherent memory
        size += context->nonCoherentAtomSize - (size % context->nonCoherentAtomSize);
        size = std::min(size, bufferSize);
    }
    return size;
}

static NnByte *findFirstOpConfig(NnNodeConfig *nodeConfig, NnOpCode opCode) {
    for (NnUint i = 0; i < nodeConfig->nSegments; i++) {
        NnSegmentConfig *segmentConfig = &nodeConfig->segments[i];
        for (NnUint j = 0; j < segmentConfig->nOps; j++) {
            if (segmentConfig->ops[j].code == opCode)
                return segmentConfig->ops[j].config;
        }
    }
    return nullptr;
}

NnVulkanDeviceData::NnVulkanDeviceData(NnVulkanContext *context, NnNetConfig *netConfig, NnNodeConfig *nodeConfig) :
    pipes(netConfig->nPipes),
    buffers(nodeConfig->nBuffers),
    internalBuffers()
{
    this->netConfig = netConfig;
    this->nodeConfig = nodeConfig;

    for (NnUint i = 0; i < netConfig->nPipes; i++) {
        const NnPipeConfig *pipeConfig = &netConfig->pipes[i];
        const bool isSliceable = pipeConfig->size.z == 1u;
        pipes[i].reset(new NnVulkanBuffer(context, netConfig->pipes[i].name, pipeConfig->size.nBytes, isSliceable, vk::BufferUsageFlagBits::eStorageBuffer, true));
    }
    for (NnUint i = 0; i < nodeConfig->nBuffers; i++) {
        const NnBufferConfig *config = &nodeConfig->buffers[i];
        const bool isSliceable = config->size.z == 1u;
        buffers[i].reset(new NnVulkanBuffer(context, nodeConfig->buffers[i].name, config->size.nBytes, isSliceable, vk::BufferUsageFlagBits::eStorageBuffer, false));
    }

    NnRopeOpConfig *ropeLlamaOpConfig = (NnRopeOpConfig *)findFirstOpConfig(nodeConfig, OP_ROPE);
    if (ropeLlamaOpConfig != nullptr) {
        assert(ropeLlamaOpConfig->ropeCacheBufferIndex < nodeConfig->nBuffers);
        NnVulkanBuffer *buffer = buffers[ropeLlamaOpConfig->ropeCacheBufferIndex].get();
        std::vector<NnByte> ropeCache(ropeLlamaOpConfig->slice.cacheSize.nBytes);
        fullfillRopeCache(ropeLlamaOpConfig, (float *)ropeCache.data());
        buffer->write(ropeCache.data());
    }
}

NnVulkanDeviceData::~NnVulkanDeviceData() {
    pipes.clear();
    buffers.clear();
    internalBuffers.clear();
}

NnSize3D NnVulkanDeviceData::resolveBufferSize(NnPointerConfig *config) {
    if (config->source == SRC_BUFFER)
        return nodeConfig->buffers[config->pointerIndex].size;
    if (config->source == SRC_PIPE)
        return netConfig->pipes[config->pointerIndex].size;
    throw std::invalid_argument("Unsupported pointer config");
}

NnVulkanBuffer *NnVulkanDeviceData::resolvePointerVulkanBuffer(NnPointerConfig *config) {
    if (config->source == SRC_BUFFER)
        return buffers[config->pointerIndex].get();
    if (config->source == SRC_PIPE)
        return pipes[config->pointerIndex].get();
    throw std::invalid_argument("Unsupported pointer config");
}

NnUint NnVulkanDeviceData::resolveBufferBatchOffset(NnPointerConfig *config, NnUint batchIndex, NnUint zIndex) {
    assert(batchIndex < netConfig->nBatches);
    if (config->type == PNTR_RAW)
        return 0;

    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint offsetZ = bufferSize.x * netConfig->nBatches * zIndex;
    const NnUint sizeX = bufferSize.x / blockSize;

    if (config->type == PNTR_BATCH)
        return offsetZ + sizeX * batchIndex;
    if (config->type == PNTR_BATCHED_SLICE) {
        assert(sizeX % netConfig->nNodes == 0);
        return offsetZ + sizeX * batchIndex + (sizeX / netConfig->nNodes) * nodeConfig->nodeIndex;
    }
    throw std::runtime_error("Cannot determine buffer offset");
}

NnUint NnVulkanDeviceData::resolveBufferBatchWidth(NnPointerConfig *config) {
    const NnSize3D bufferSize = resolveBufferSize(config);
    const NnSize blockSize = getBlockSize(bufferSize.floatType);
    assert(bufferSize.x % blockSize == 0);
    const NnUint sizeX = bufferSize.x / blockSize;

    if (config->type == PNTR_RAW)
        return sizeX;
    if (config->type == PNTR_BATCH)
        return sizeX;
    if (config->type == PNTR_BATCHED_SLICE) {
        assert(sizeX % netConfig->nNodes == 0);
        return sizeX / netConfig->nNodes;
    }
    throw std::runtime_error("Cannot determine buffer width");
}

NnVulkanDevice::NnVulkanDevice(NnUint gpuIndex, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
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
    deviceExtension.push_back("VK_KHR_8bit_storage");
    deviceExtension.push_back("VK_KHR_16bit_storage");
    deviceExtension.push_back("VK_KHR_shader_float16_int8");
    deviceExtension.push_back("VK_KHR_maintenance4");

    vk::ApplicationInfo appInfo {"Distributed Llama", 1, nullptr, 0, VK_API_VERSION_1_2 };
    vk::InstanceCreateInfo instanceCreateInfo(createInstanceFlags, &appInfo, instanceLayers, instanceExtensions);
    context.instance = vk::createInstance(instanceCreateInfo);

    auto physicalDevices = context.instance.enumeratePhysicalDevices();
    const NnSize nDevices = physicalDevices.size();
    if (gpuIndex >= nDevices)
        throw std::runtime_error("Invalid GPU index, found " + std::to_string(nDevices) + " GPUs");
    context.physicalDevice = physicalDevices[gpuIndex];
    assertDeviceExtensionSupport(context.physicalDevice, deviceExtension);

    vk::PhysicalDeviceProperties deviceProps = context.physicalDevice.getProperties();
    printf("🌋 Device: %s\n", (char*)deviceProps.deviceName);
    printf("🌋 DeviceApiVersion: %d.%d.%d\n", VK_VERSION_MAJOR(deviceProps.apiVersion), VK_VERSION_MINOR(deviceProps.apiVersion), VK_VERSION_PATCH(deviceProps.apiVersion));
    printf("🌋 MaxComputeSharedMemory: %d kB\n", deviceProps.limits.maxComputeSharedMemorySize / 1024);
    printf("🌋 NonCoherentAtomSize: %lu bytes\n", (NnSize)deviceProps.limits.nonCoherentAtomSize);

    vk::PhysicalDeviceMemoryProperties memoryProperties = context.physicalDevice.getMemoryProperties();
    for (unsigned int h = 0; h < memoryProperties.memoryHeapCount; h++) {
        if (memoryProperties.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
            printf("🌋 Heap[%u]: %lu MB\n", h, ((NnSize)memoryProperties.memoryHeaps[h].size) / (1024 * 1024));
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

    vk::CommandPoolCreateInfo commandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer), context.queueFamilyIndex);
    context.commandPool = context.device.createCommandPool(commandPoolCreateInfo);
    context.queue = context.device.getQueue(context.queueFamilyIndex, 0);
    context.nonCoherentAtomSize = deviceProps.limits.nonCoherentAtomSize;

    VULKAN_TRACE("Context created");
    data = new NnVulkanDeviceData(&context, netConfig, nodeConfig);
}

NnVulkanDevice::~NnVulkanDevice() {
    delete data;

    context.device.destroyCommandPool(context.commandPool);
    context.device.destroy();
    context.instance.destroy();
    VULKAN_TRACE("Destroyed device");
}

NnUint NnVulkanDevice::maxNThreads() {
    return 1;
}

NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    return new NnVulkanDeviceSegment(&context, data, netConfig, segmentIndex, segmentConfig, netExecution);
};

static const char *getShaderFileName(const NnOpCode opCode, const NnOpQuantType quantType) {
    if (opCode == OP_MERGE_ADD) {
        if (quantType == F32_F32_F32) return "merge-add-forward-f32-f32.spv";
        if (quantType == Q80_Q80_F32) return "merge-add-forward-q80-f32.spv";
    }
    if (opCode == OP_MERGE_SUM) {
        if (quantType == F32_F32_F32) return "merge-sum-forward-f32-f32.spv";
    }
    if (opCode == OP_EMBEDDING) {
        if (quantType == F32_F32_F32) return "embedding-forward-f32-f32.spv";
    }
    if (opCode == OP_ROPE) {
        if (quantType == F32_F32_F32) return "rope-forward-f32-f32.spv";
    }
    if (opCode == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return "inv-rms-forward-f32-f32.spv";
    }
    if (opCode == OP_MATMUL) {
        if (quantType == F32_F32_F32) return "matmul-forward-f32-f32-f32.spv";
        if (quantType == Q80_Q40_F32) return "matmul-forward-q80-q40-f32.spv";
    }
    if (opCode == OP_MULTIHEAD_ATT) {
        if (quantType == F32_F32_F32) return "multi-head-att-forward-f32-f32.spv";
    }
    if (opCode == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return "rms-norm-forward-f32-f32-f32.spv";
    }
    if (opCode == OP_SILU) {
        if (quantType == F32_F32_F32) return "silu-forward-f32-f32.spv";
    }
    if (opCode == OP_MUL) {
        if (quantType == F32_F32_F32) return "mul-forward-f32-f32.spv";
    }
    if (opCode == OP_SCALE) {
        if (quantType == F32_F32_F32) return "scale-forward-f32-f32.spv";
    }
    if (opCode == OP_CAST) {
        if (quantType == F32_F32_F32) return "cast-forward-f32-f32.spv";
        if (quantType == F32_F32_Q80) return "cast-forward-f32-q80.spv";
    }
    if (opCode == OP_REPEAT_Z) {
        if (quantType == F32_F32_Q80) return "repeatz-forward-f32-q80.spv";
    }
    if (opCode == OP_SHIFT) {
        if (quantType == F32_F32_F32) return "shift-forward-f32-f32.spv";
    }
    if (opCode == OP_SOFTMAX) {
        if (quantType == F32_F32_F32) return "softmax-forward-f32-f32.spv";
    }
    if (opCode == OP_MOE_GATE) {
        if (quantType == F32_F32_F32) return "moe-gate-forward-f32-f32.spv";
    }
    throw std::invalid_argument(std::string("Unsupported shader: ") + opCodeToString(opCode) + "/" + opQuantTypeToString(quantType));
}

static void buildShaderLayout(std::vector<NnOpBufferAccess> &a, NnVulkanDeviceData *data, NnVulkanDeviceSegmentData *segmentData, NnUint opIndex, NnOpConfig *opConfig) {
    // input
    a.push_back({ACCESS_READONLY, data->resolvePointerVulkanBuffer(&opConfig->input)});
    // output
    a.push_back({ACCESS_READ_WRITE, data->resolvePointerVulkanBuffer(&opConfig->output)});
    // batch info
    a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpBatchInfoVulkanBuffer(opIndex)});
    // weight
    if (opConfig->weightSize.nBytes > 0)
        a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpWeightVulkanBuffer(opIndex)});
    // config
    if (opConfig->configSize > 0) {
        a.push_back({ACCESS_IMMUTABLE, segmentData->resolveOpConfigVulkanBuffer(opIndex)});

        switch (opConfig->code) {
        case OP_RMS_NORM: {
            const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->buffers[config->invRmsBufferIndex].get()});
        } break;
        case OP_MUL: {
            const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->buffers[config->multiplierBufferIndex].get()});
        } break;
        case OP_MATMUL: {
            const NnMatmulOpConfig *config = (NnMatmulOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->buffers[config->activeExpertIndexesBufferIndex].get()});
        } break;
        case OP_SCALE: {
            const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->buffers[config->scaleBufferIndex].get()});
        } break;
        case OP_SHIFT: {
            const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->pipes[config->indexPipeIndex].get()});
        } break;
        case OP_ROPE: {
            const NnRopeOpConfig *config = (NnRopeOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->pipes[config->positionPipeIndex].get()});
            a.push_back({ACCESS_IMMUTABLE, data->buffers[config->ropeCacheBufferIndex].get()});
        } break;
        case OP_MULTIHEAD_ATT: {
            const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)opConfig->config;
            a.push_back({ACCESS_READONLY, data->pipes[config->positionPipeIndex].get()});
            a.push_back({ACCESS_READONLY, data->buffers[config->queryBufferIndex].get()});
            a.push_back({ACCESS_READONLY, data->buffers[config->keyCacheBufferIndex].get()});
            a.push_back({ACCESS_READONLY, data->buffers[config->valueCacheBufferIndex].get()});
            a.push_back({ACCESS_READ_WRITE, data->buffers[config->attBufferIndex].get()});
        } break;
        case OP_MOE_GATE: {
            const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)opConfig->config;
            a.push_back({ACCESS_READ_WRITE, data->buffers[config->indexesBufferIndex].get()});
        } break;
        default:
            break;
        }
    }
}

static bool containsBuffer(std::vector<NnVulkanBuffer *> &buffers, NnVulkanBuffer *buffer) {
    for (NnVulkanBuffer *b : buffers) {
        if (b == buffer) return true;
    }
    return false;
}

static void resolveBuffersToSync(std::vector<std::vector<NnVulkanBuffer *>>& buffersToSync, std::vector<std::vector<NnOpBufferAccess>>& accesses) {
    std::vector<NnVulkanBuffer *> modifiedBuffers;
    for (NnUint opIndex = 0; opIndex < accesses.size(); opIndex++) {
        if (opIndex > 0) {
            bool flush = false;
            for (const NnOpBufferAccess &access : accesses[opIndex]) {
                if (access.type == ACCESS_READONLY && containsBuffer(modifiedBuffers, access.buffer)) {
                    flush = true;
                    break;
                }
            }
            if (flush) {
                for (NnVulkanBuffer *buffer : modifiedBuffers)
                    buffersToSync[opIndex].push_back(buffer);
                modifiedBuffers.clear();
            }
        }
        for (const NnOpBufferAccess &access : accesses[opIndex]) {
            if (access.type == ACCESS_READ_WRITE && !containsBuffer(modifiedBuffers, access.buffer))
                modifiedBuffers.push_back(access.buffer);
        }
    }
}

static NnUint resolveNumberOfBatchInfoZ(const NnSize3D inputSize, const NnSize3D outputSize) {
    return std::max(inputSize.z, outputSize.z);
}

static std::vector<NnVulkanBatchInfo> buildBatchInfo(NnOpConfig *opConfig, NnVulkanDeviceData *data, NnUint nBatches) {
    const NnUint nZ = resolveNumberOfBatchInfoZ(
        data->resolveBufferSize(&opConfig->input),
        data->resolveBufferSize(&opConfig->output)
    );

    std::vector<NnVulkanBatchInfo> batchInfo(nZ * nBatches);
    for (NnUint zIndex = 0; zIndex < nZ; zIndex++) {
        for (NnUint batchIndex = 0; batchIndex < nBatches; batchIndex++) {
            const NnUint i = zIndex * nBatches + batchIndex;
            batchInfo[i].inputOffset = data->resolveBufferBatchOffset(&opConfig->input, batchIndex, zIndex);
            batchInfo[i].inputSizeX = data->resolveBufferBatchWidth(&opConfig->input);
            batchInfo[i].outputOffset = data->resolveBufferBatchOffset(&opConfig->output, batchIndex, zIndex);
            batchInfo[i].outputSizeX = data->resolveBufferBatchWidth(&opConfig->output);
        }
    }
    return batchInfo;
}

static std::vector<NnUint> resolveShaderConstants(const NnOpConfig *opConfig, const NnUint nBatches, const NnSize3D inputSize, const NnSize3D outputSize) {
    std::vector<NnUint> consts;
    consts.push_back(nBatches);

    if (
        opConfig->code == OP_MATMUL ||
        opConfig->code == OP_MERGE_SUM ||
        opConfig->code == OP_MUL ||
        opConfig->code == OP_REPEAT_Z ||
        opConfig->code == OP_SCALE ||
        opConfig->code == OP_SILU ||
        opConfig->code == OP_SOFTMAX
    ) {
        const NnUint nZ = resolveNumberOfBatchInfoZ(inputSize, outputSize);
        consts.push_back(nZ);
    }
    if (opConfig->code == OP_MATMUL) {
        if (opConfig->weightSize.floatType == F_Q40) {
            constexpr NnUint tileSizeX = 2; // Shader constant
            assert(inputSize.x % (Q40_BLOCK_SIZE * tileSizeX) == 0);
            consts.push_back(inputSize.x / (Q40_BLOCK_SIZE * tileSizeX));
        }
    }
    if (opConfig->code == OP_MOE_GATE) {
        const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)opConfig->config;
        consts.push_back(inputSize.x);
        consts.push_back(config->k);
    }
    return consts;
}

static NnUint resolveShaderNumberOfWorkGroupsX(const NnOpConfig *opConfig, const NnSize3D inputSize, const NnSize3D outputSize) {
    if (
        opConfig->code == OP_CAST ||
        opConfig->code == OP_REPEAT_Z
    ) {
        if (outputSize.floatType == F_Q80) {
            return outputSize.x / Q80_BLOCK_SIZE;
        } else {
            constexpr NnUint chunkSize = 4u; // Shader constant
            return outputSize.x / chunkSize;
        }
    }
    if (opConfig->code == OP_MERGE_ADD) {
        if (inputSize.floatType == F_Q80) {
            return outputSize.x / Q80_BLOCK_SIZE; // Yes, outputSize is used here
        } else {
            return 32u;
        }
    }
    if (opConfig->code == OP_MATMUL) {
        if (opConfig->weightSize.floatType == F_Q40) {
            constexpr NnUint tileSizeD = 8u; // Shader constant
            assert(opConfig->weightSize.x % tileSizeD == 0u);
            return opConfig->weightSize.x / tileSizeD;
        } else {
            return 32u;
        }
    }
    if (opConfig->code == OP_MULTIHEAD_ATT)
        return ((NnMultiHeadAttOpConfig *)opConfig->config)->nHeads0;
    if (opConfig->code == OP_INV_RMS)
        return ((NnInvRmsOpConfig *)opConfig->config)->nColumns;
    if (
        opConfig->code == OP_EMBEDDING ||
        opConfig->code == OP_RMS_NORM ||
        opConfig->code == OP_MUL ||
        opConfig->code == OP_SCALE ||
        opConfig->code == OP_SILU ||
        opConfig->code == OP_SHIFT ||
        opConfig->code == OP_MERGE_SUM
    ) {
        constexpr NnUint chunkSize = 4u; // Shader constant
        assert(outputSize.x % chunkSize == 0);
        return outputSize.x / chunkSize;
    }
    return 1u;
}

static NnUint resolveShaderNumberOfWorkGroupsZ(const NnOpConfig *opConfig, const NnSize3D inputSize, const NnSize3D outputSize) {
    if (
        opConfig->code == OP_MERGE_SUM ||
        opConfig->code == OP_REPEAT_Z
    ) {
        return 1u;
    }
    return resolveNumberOfBatchInfoZ(inputSize, outputSize);
}

static std::vector<uint32_t> readShader(const char *fileName) {
    std::vector<uint32_t> code;
    std::string path = std::string("./src/nn/vulkan/") + fileName;
    FILE *file = fopen(path.c_str(), "rb");
    if (!file)
        throw std::runtime_error("Failed to open shader file: " + path);
    constexpr size_t maxSize = 16384;
    uint32_t chunk[maxSize];
    size_t bytesRead = fread(chunk, 1, maxSize, file);
    assert(bytesRead < maxSize); // Check if the file is too large
    if (bytesRead > 0)
        code.insert(code.end(), chunk, chunk + bytesRead);
    if (ferror(file)) {
        fclose(file);
        throw std::runtime_error("Failed to read shader file: " + path);
    }
    fclose(file);
    return code;
}

static vk::DescriptorType toDescriptorType(NnVulkanBuffer *buffer) {
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eUniformBuffer)
        return vk::DescriptorType::eUniformBuffer;
    if (buffer->usageFlags & vk::BufferUsageFlagBits::eStorageBuffer)
        return vk::DescriptorType::eStorageBuffer;
    throw std::invalid_argument("Unsupported buffer usage");
}

NnVulkanDeviceSegmentData::NnVulkanDeviceSegmentData(NnVulkanContext *context, NnVulkanDeviceData *data, NnSegmentConfig *segmentConfig, NnUint nBatches) :
    batchInfoBufferIndex(segmentConfig->nOps, UINT32_MAX),
    weightBufferIndex(segmentConfig->nOps, UINT32_MAX),
    configBufferIndex(segmentConfig->nOps, UINT32_MAX)
{
    this->data = data;

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];

        std::vector<NnVulkanBatchInfo> batchInfo = buildBatchInfo(opConfig, data, nBatches);
        NnSize batchInfoSize = sizeof(NnVulkanBatchInfo) * batchInfo.size();
        NnVulkanBuffer *batchInfoBuffer = new NnVulkanBuffer(context, "batchInfo", batchInfoSize, false, vk::BufferUsageFlagBits::eUniformBuffer, false);
        data->internalBuffers.push_back(std::unique_ptr<NnVulkanBuffer>(batchInfoBuffer));
        batchInfoBuffer->write((NnByte *)batchInfo.data());
        batchInfoBufferIndex[opIndex] = data->internalBuffers.size() - 1;

        if (opConfig->weightSize.nBytes > 0) {
            NnVulkanBuffer *buffer = new NnVulkanBuffer(context, "weights", opConfig->weightSize.nBytes, false, vk::BufferUsageFlagBits::eStorageBuffer, false);
            data->internalBuffers.push_back(std::unique_ptr<NnVulkanBuffer>(buffer));
            weightBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
        if (opConfig->configSize > 0) {
            NnVulkanBuffer *configBuffer = new NnVulkanBuffer(context, "config", opConfig->configSize, false, vk::BufferUsageFlagBits::eUniformBuffer, false);
            data->internalBuffers.push_back(std::unique_ptr<NnVulkanBuffer>(configBuffer));
            configBuffer->write(opConfig->config);
            configBufferIndex[opIndex] = data->internalBuffers.size() - 1;
        }
    }
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpBatchInfoVulkanBuffer(NnUint opIndex) {
    assert(batchInfoBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[batchInfoBufferIndex[opIndex]].get();
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpConfigVulkanBuffer(NnUint opIndex) {
    assert(configBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[configBufferIndex[opIndex]].get();
}

NnVulkanBuffer *NnVulkanDeviceSegmentData::resolveOpWeightVulkanBuffer(NnUint opIndex) {
    assert(weightBufferIndex[opIndex] != UINT32_MAX);
    return data->internalBuffers[weightBufferIndex[opIndex]].get();
}

NnVulkanDeviceSegment::NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanDeviceData *data, NnNetConfig *netConfig, NnUint segmentIndex, NnSegmentConfig *segmentConfig, NnNetExecution *netExecution) :
    shaderModules(segmentConfig->nOps),
    descriptorSets(segmentConfig->nOps),
    descriptorSetLayouts(segmentConfig->nOps),
    pipelineLayouts(segmentConfig->nOps),
    pipelines(segmentConfig->nOps),
    buffersToSync(segmentConfig->nOps)
{
    this->context = context;
    this->data = data;
    this->netConfig = netConfig;
    this->segmentIndex = segmentIndex;
    this->segmentConfig = segmentConfig;
    this->netExecution = netExecution;
    this->segmentData.reset(new NnVulkanDeviceSegmentData(context, data, segmentConfig, netExecution->nBatches));
    this->lastBatchSize = 0;

    std::vector<vk::PipelineShaderStageCreateInfo> shaderCreateInfos(segmentConfig->nOps);
    std::vector<std::vector<NnOpBufferAccess>> opBufferAccesses(segmentConfig->nOps);

    std::vector<vk::SpecializationInfo> specInfos(segmentConfig->nOps);
    std::vector<std::vector<vk::SpecializationMapEntry>> specEntries(segmentConfig->nOps);
    std::vector<std::vector<NnUint>> constants(segmentConfig->nOps);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        NnSize3D inputSize = data->resolveBufferSize(&opConfig->input);
        NnSize3D outputSize = data->resolveBufferSize(&opConfig->output);
        NnOpQuantType opQuant = getOpQuantType(
            inputSize.floatType,
            opConfig->weightSize.floatType,
            outputSize.floatType
        );
        const char *shaderFileName = getShaderFileName(opConfig->code, opQuant);
        std::vector<uint32_t> code = readShader(shaderFileName);

        buildShaderLayout(opBufferAccesses[opIndex], data, segmentData.get(), opIndex, opConfig);

        VULKAN_TRACE("Loading shader: %s", shaderFileName);
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            code.size(),
            code.data()
        );

        constants[opIndex] = resolveShaderConstants(opConfig, netConfig->nBatches, inputSize, outputSize);
        const std::vector<NnUint> *opConstants = &constants[opIndex];

        specEntries[opIndex].resize(opConstants->size());
        for (NnUint i = 0; i < opConstants->size(); i++)
            specEntries[opIndex][i] = vk::SpecializationMapEntry(
                i,
                i * sizeof(NnUint),
                sizeof(NnUint)
            );
        specInfos[opIndex] = vk::SpecializationInfo(
            opConstants->size(),
            specEntries[opIndex].data(),
            opConstants->size() * sizeof(NnUint),
            opConstants->data()
        );

        vk::ShaderModule shaderModule = context->device.createShaderModule(shaderModuleCreateInfo);
        vk::PipelineShaderStageCreateInfo shaderCreateInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            shaderModule,
            "main",
            &specInfos[opIndex]
        );

        shaderModules[opIndex] = shaderModule;
        shaderCreateInfos[opIndex] = shaderCreateInfo;
        VULKAN_TRACE("Segment %d, opIndex: %d, buffers: %zu", segmentIndex, opIndex, buffers.size());
    }

    NnUint nUniformBuffers = 0;
    NnUint nStorageBuffers = 0;
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings(accesses.size());
        for (NnUint i = 0; i < accesses.size(); i++) {
            vk::DescriptorType descriptorType = toDescriptorType(accesses[i].buffer);
            descriptorSetLayoutBindings[i] = vk::DescriptorSetLayoutBinding(
                i,
                descriptorType,
                1,
                vk::ShaderStageFlagBits::eCompute
            );
            if (descriptorType == vk::DescriptorType::eUniformBuffer)
                nUniformBuffers++;
            else if (descriptorType == vk::DescriptorType::eStorageBuffer)
                nStorageBuffers++;
        }

        vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            descriptorSetLayoutBindings.size(),
            descriptorSetLayoutBindings.data()
        );
        vk::DescriptorSetLayout descriptorSetLayout = context->device.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);

        descriptorSetLayouts[opIndex] = descriptorSetLayout;
    }

    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes;
    if (nStorageBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, nStorageBuffers));
    if (nUniformBuffers > 0)
        descriptorPoolSizes.push_back(vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, nUniformBuffers));

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 
        segmentConfig->nOps,
        descriptorPoolSizes.size(),
        descriptorPoolSizes.data()
    );
    descriptorPool = context->device.createDescriptorPool(descriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo(descriptorPool, descriptorSetLayouts.size(), descriptorSetLayouts.data());
    descriptorSets = context->device.allocateDescriptorSets(descriptorSetAllocInfo);

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets(nUniformBuffers + nStorageBuffers);
    std::vector<vk::DescriptorBufferInfo> bufferInfos(nUniformBuffers + nStorageBuffers);
    NnUint writeDescriptorSetIndex = 0;
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        std::vector<NnOpBufferAccess> &accesses = opBufferAccesses[opIndex];
        for (NnUint i = 0; i < accesses.size(); i++) {
            NnVulkanBuffer *buffer = accesses[i].buffer;
            bufferInfos[writeDescriptorSetIndex] = vk::DescriptorBufferInfo(
                buffer->deviceBuffer,
                0,
                buffer->bufferSize
            );
            vk::DescriptorType descriptorType = toDescriptorType(buffer);
            writeDescriptorSets[writeDescriptorSetIndex] = vk::WriteDescriptorSet(
                descriptorSets[opIndex],
                i,
                0,
                1,
                descriptorType,
                nullptr,
                &bufferInfos.data()[writeDescriptorSetIndex],
                nullptr
            );
            writeDescriptorSetIndex++;
        }
    }

    context->device.updateDescriptorSets(writeDescriptorSets, nullptr);

    pipelineCache = context->device.createPipelineCache(vk::PipelineCacheCreateInfo());

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1, &descriptorSetLayouts[opIndex]);
        pipelineLayouts[opIndex] = context->device.createPipelineLayout(pipelineLayoutCreateInfo);

        vk::ComputePipelineCreateInfo pipelineInfo(vk::PipelineCreateFlags(), shaderCreateInfos[opIndex], pipelineLayouts[opIndex], vk::Pipeline(), 0);
        pipelines[opIndex] = context->device.createComputePipelines(pipelineCache, pipelineInfo).value.front();
    }

    fence = context->device.createFence(vk::FenceCreateInfo());

    vk::CommandBufferAllocateInfo commandBufferAllocInfo(context->commandPool, vk::CommandBufferLevel::ePrimary, 1);
    const std::vector<vk::CommandBuffer> cmdBuffers = context->device.allocateCommandBuffers(commandBufferAllocInfo);
    commandBuffer = cmdBuffers.front();

    resolveBuffersToSync(buffersToSync, opBufferAccesses);
}

NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
    context->device.freeCommandBuffers(context->commandPool, 1, &commandBuffer);
    context->device.destroyDescriptorPool(descriptorPool);

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        context->device.destroyPipeline(pipelines[opIndex]);
        context->device.destroyPipelineLayout(pipelineLayouts[opIndex]);
    }
    context->device.destroyFence(fence);
    context->device.destroyPipelineCache(pipelineCache);
    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        context->device.destroyDescriptorSetLayout(descriptorSetLayouts[opIndex]);
        context->device.destroyShaderModule(shaderModules[opIndex]);
    }
    VULKAN_TRACE("Destroyed segment");
}

void NnVulkanDeviceSegment::loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) {
    assert(opIndex < segmentConfig->nOps);
    assert(offset + nBytes <= segmentConfig->ops[opIndex].weightSize.nBytes);
    NnVulkanBuffer *buffer = segmentData->resolveOpWeightVulkanBuffer(opIndex);
    buffer->write(weight, offset, nBytes);
}

void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    assert(threadIndex == 0);
    if (opIndex != 0) {
        // TODO: this is a design problem, executor tries to forward all ops in a segment
        return;
    }

    // TODO: refactor this block
    {
        if (segmentIndex == 0 || segmentIndex == 1) { // TODO: this is a hack to fix workers
            for (NnUint i = 0; i < netConfig->nPreSyncs; i++) {
                NnPreSyncConfig *preSyncConfig = &netConfig->preSyncs[i];
                NnByte *pipeData = netExecution->pipes[preSyncConfig->pipeIndex];
                NnVulkanBuffer *buffer = data->pipes[preSyncConfig->pipeIndex].get();
                buffer->write(pipeData, 0u, buffer->calcSliceSize(batchSize, netConfig->nBatches));
            }
        }

        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->input.source == SRC_PIPE) {
                NnByte *pipeData = netExecution->pipes[opConfig->input.pointerIndex];
                NnVulkanBuffer *buffer = data->pipes[opConfig->input.pointerIndex].get();
                buffer->write(pipeData, 0u, buffer->calcSliceSize(batchSize, netConfig->nBatches));
            }
        }
    }

    if (lastBatchSize != batchSize) {
        lastBatchSize = batchSize;
        commandBuffer.begin({ vk::CommandBufferUsageFlags{} });

        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            std::vector<vk::BufferMemoryBarrier> memoryBarriers;
            for (NnVulkanBuffer *buffer : buffersToSync[opIndex]) {
                vk::BufferMemoryBarrier barrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead,
                    context->queueFamilyIndex,
                    context->queueFamilyIndex,
                    buffer->deviceBuffer,
                    0,
                    buffer->calcSliceSize(batchSize, netConfig->nBatches)
                );
                memoryBarriers.push_back(barrier);
            }
            if (!memoryBarriers.empty()) {
                commandBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(),
                    nullptr,
                    memoryBarriers,
                    nullptr
                );
                VULKAN_TRACE("Created memory barrier for %zu buffers", barriers.size());
            }

            commandBuffer.bindPipeline(
                vk::PipelineBindPoint::eCompute,
                pipelines[opIndex]
            );
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute,
                pipelineLayouts[opIndex],
                0,
                { descriptorSets[opIndex] },
                {}
            );

            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            const NnSize3D inputSize = data->resolveBufferSize(&opConfig->input);
            const NnSize3D outputSize = data->resolveBufferSize(&opConfig->output);

            const NnUint groupCountX = resolveShaderNumberOfWorkGroupsX(opConfig, inputSize, outputSize);
            const NnUint groupCountY = batchSize;
            const NnUint groupCountZ = resolveShaderNumberOfWorkGroupsZ(opConfig, inputSize, outputSize);
            commandBuffer.dispatch(groupCountX, groupCountY, groupCountZ);
            VULKAN_TRACE("Dispatched %d %d %d", groupCountX, groupCountY, groupCountZ);
        }
        commandBuffer.end();
    }

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    context->queue.submit({ submitInfo }, fence);
    assert(context->device.waitForFences({ fence }, true, uint64_t(-1)) == vk::Result::eSuccess);

    context->device.resetFences({ fence });
    VULKAN_TRACE("Forwarded");

    // TODO: refactor this block
    {
        for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
            NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
            if (opConfig->output.source == SRC_PIPE) {
                NnByte *pipeData = netExecution->pipes[opConfig->output.pointerIndex];
                NnVulkanBuffer *buffer = data->pipes[opConfig->output.pointerIndex].get();
                buffer->read(pipeData, 0u, buffer->calcSliceSize(batchSize, netConfig->nBatches));
            }
        }
    }

#if DEBUG_VULKAN_BUFFERS
    NnUint nBuffers = data->buffers.size();
    for (NnUint i = 0; i < nBuffers; i++) {
        NnVulkanBuffer *buffer = data->buffers[i].get();
        printf("[%3d:%3d:%10s] ", segmentIndex, i, buffer->name);
        std::vector<NnByte> data(buffer->bufferSize);
        buffer->read(data.data());
        if (strncmp(buffer->name, "q_", 2) == 0) {
            NnUint nBytes = 32;
            if (buffer->bufferSize < nBytes)
                nBytes = buffer->bufferSize;
            for (NnUint j = 0; j < nBytes; j++)
                printf(" %x", data.data()[j]);
        } else {
            NnUint nNumbers = buffer->bufferSize / sizeof(float);
            if (nNumbers > 16)
                nNumbers = 16;
            float *nums = (float *)data.data();
            for (NnUint j = 0; j < nNumbers; j++)
                printf(" %.4f", nums[j]);
        }
        printf("\n");
    }
#endif
}
