#ifndef NN_VULKAN_HPP
#define NN_VULKAN_HPP

#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

class NnVulkanDevice : public NnDevice {
public:
    NnVulkanDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnVulkanDevice() override;
    NnUint maxNThreads() override;
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;
};

class NnVulkanDeviceSegment : public NnDeviceSegment {
public:
    NnVulkanDeviceSegment();
    ~NnVulkanDeviceSegment() override;
    void loadWeight(NnUint opIndex, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif