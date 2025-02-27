#include <cstdio>
#include "nn-config-builder.hpp"
#include "nn-vulkan.hpp"

#define N_BATCHES 4

void printOk(const char *name) {
    printf("✅ %24s passed\n", name);
}

void execute(
    void (*build)(NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder),
    void (*execute)(NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device)
) {
    NnUint nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, N_BATCHES);
    NnNodeConfigBuilder nodeBuilder(0);
    NnSegmentConfigBuilder segmentBuilder;
    build(&netBuilder, &nodeBuilder, &segmentBuilder);
    nodeBuilder.addSegment(segmentBuilder.build());

    NnNetConfig netConfig = netBuilder.build();
    NnNodeConfig nodeConfig = nodeBuilder.build();
    std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
    std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

    NnNetExecution execution(1, &netConfig);
    execution.setBatchSize(1);

    NnVulkanDevice device(&netConfig, &nodeConfig, &execution);
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&netConfig, &nodeConfig, &device, &execution, &synchronizer);

    execute(&executor, &execution, &device);
}

void testInvRms_F32_F32() {
    #define INV_RMS_DIM 256
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, INV_RMS_DIM));
            NnUint invRmsBufferIndex = nodeBuilder->addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1));
            segmentBuilder->addOp(OP_INV_RMS, "inv_rms", 0,
                pointerConfig(PNTR_PIPE, xPipeIndex),
                pointerConfig(PNTR_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{1e-5f});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            float *xPipe = (float *)execution->pipes[0];
            for (NnUint i = 0; i < INV_RMS_DIM; i++)
                xPipe[i] = i / (float)INV_RMS_DIM;

            executor->forward();

            float result[N_BATCHES];
            device->data->buffers[0].get()->read((NnByte *)result);

            float expectedValue = 1.737115f;
            float diff = fabs(expectedValue - result[0]);
            printf("expectedValue: %f, result: %f, diff: %f\n", expectedValue, result[0], diff);
            assert(diff < 0.0001f);
            printOk("testInvRms_F32_F32");
        });
}

int main() {
    testInvRms_F32_F32();
    return 0;
}
