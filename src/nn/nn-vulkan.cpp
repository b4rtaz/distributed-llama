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

void NnVulkanStagingCopy::addCopyCommand(vk::CommandBuffer commandBuffer) {
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
}

NnVulkanData::NnVulkanData(const NnUint nPipes, const NnUint nBuffers)
    : pipes(nPipes), buffers(nBuffers) {}

NnVulkanDevice::NnVulkanDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution)
    : data(netConfig->nPipes, nodeConfig->nBuffers)
{
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

    for (NnUint i = 0; i < netConfig->nPipes; i++)
        data.pipes[i].reset(new NnVulkanBuffer(&context, netConfig->pipes[i].size.nBytes, vk::BufferUsageFlagBits::eUniformBuffer, true));
    for (NnUint i = 0; i < nodeConfig->nBuffers; i++)
        data.buffers[i].reset(new NnVulkanBuffer(&context, nodeConfig->buffers[i].size.nBytes, vk::BufferUsageFlagBits::eUniformBuffer, false));
}

NnVulkanDevice::~NnVulkanDevice() {
    data.pipes.clear();
    data.buffers.clear();
    context.device.destroyCommandPool(context.commandPool);
    context.device.destroy();
    context.instance.destroy();
}

NnUint NnVulkanDevice::maxNThreads() {
    return 1;
}

NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    NnSegmentConfig *segmentConfig = &nodeConfig->segments[segmentIndex];
    return new NnVulkanDeviceSegment(&context, &data, segmentConfig);
};

void NnVulkanDevice::syncPointers() {
}

NnVulkanDeviceSegment::NnVulkanDeviceSegment(NnVulkanContext *context, NnVulkanData *data, NnSegmentConfig *segmentConfig)
    : weightBufferIndex(segmentConfig->nOps) {
    this->context = context;
    this->data = data;
    this->segmentConfig = segmentConfig;

    for (NnUint opIndex = 0; opIndex < segmentConfig->nOps; opIndex++) {
        NnOpConfig *opConfig = &segmentConfig->ops[opIndex];
        if (opConfig->weightSize.nBytes > 0) {
            data->buffers.push_back(std::unique_ptr<NnVulkanBuffer>(
                new NnVulkanBuffer(context, opConfig->weightSize.nBytes, vk::BufferUsageFlagBits::eStorageBuffer, false)));
            weightBufferIndex[opIndex] = data->buffers.size() - 1;
        }
    }
}

NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
}

void NnVulkanDeviceSegment::loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) {
    assert(segmentConfig->nOps > opIndex);
    assert(segmentConfig->ops[opIndex].weightSize.nBytes == nBytes);
    NnUint dataBufferIndex = weightBufferIndex[opIndex];
    data->buffers[dataBufferIndex]->write(weight);
    VULKAN_TRACE("Loaded %ld bytes to weight buffer %d", nBytes, dataBufferIndex);
}

void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    
}
