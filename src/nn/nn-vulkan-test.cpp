#include <cstdio>
#include "nn-config-builder.hpp"
#include "nn-vulkan.hpp"

#define DIM 32
#define N_BATCHES 2

void buildConfig(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    NnUint nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, N_BATCHES);
    NnUint xPipeIndex = netBuilder.addPipe("X", size2D(F_32, N_BATCHES, DIM));

    NnNodeConfigBuilder nodeBuilder(0);
    NnUint invRmsBufferIndex = nodeBuilder.addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1));
    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addSync(xPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);

    segmentBuilder.addOp(OP_INV_RMS, "inv_rms", 0,
        pointerConfig(PNTR_PIPE, xPipeIndex),
        pointerConfig(PNTR_BUFFER, invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{1e-5f});

    segmentBuilder.addOp(OP_RMS_NORM, "rms_norm", 0,
        pointerConfig(PNTR_PIPE, xPipeIndex),
        pointerConfig(PNTR_PIPE, xPipeIndex),
        size1D(F_32, DIM),
        NnRmsNormOpConfig{invRmsBufferIndex});

    nodeBuilder.addSegment(segmentBuilder.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
}

int main() {
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    buildConfig(&netConfig, &nodeConfig);
    std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
    std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

    NnNetExecution execution(1, &netConfig);
    NnVulkanDevice device(&netConfig, &nodeConfig, &execution);
    NnFakeNodeSynchronizer synchronizer;

    float *x = (float *)execution.pipes[0];
    for (NnUint i = 0; i < DIM * N_BATCHES; i++)
        x[i] = i;

    float rmsNormWeight[DIM];
    for (NnUint i = 0; i < DIM; i++)
        rmsNormWeight[i] = 0.5 + i / (float)DIM;

    NnExecutor executor(&netConfig, &nodeConfig, &device, &execution, &synchronizer);
    executor.loadWeight("rms_norm", 0, sizeof(rmsNormWeight), (NnByte *)rmsNormWeight);

    execution.setBatchSize(N_BATCHES);
    executor.forward();
    return 0;
}
