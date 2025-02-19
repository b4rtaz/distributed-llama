#include "nn-vulkan.hpp"

NnVulkanDevice::NnVulkanDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
}

NnVulkanDevice::~NnVulkanDevice() {
}

NnUint NnVulkanDevice::maxNThreads() {
    return 1;
}

NnDeviceSegment *NnVulkanDevice::createSegment(NnUint segmentIndex) {
    return nullptr;
};

NnVulkanDeviceSegment::NnVulkanDeviceSegment() {
}

NnVulkanDeviceSegment::~NnVulkanDeviceSegment() {
}

void NnVulkanDeviceSegment::loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) {

}

void NnVulkanDeviceSegment::forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize)  {
    
}