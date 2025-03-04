#include <cstdio>
#include "nn-config-builder.hpp"
#include "nn-vulkan.hpp"

#define N_BATCHES 4

void printOk(const char *name) {
    printf("✅ %24s passed\n", name);
}

void assertFloat(const float value, const float expectedValue, const float tolerance) {
    float diff = fabs(expectedValue - value);
    if (diff > tolerance) {
        printf("❌ failed: value=%f, expectedValue=%f, diff=%f\n", value, expectedValue, diff);
        exit(1);
    }
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
    NnExecutor executor(&netConfig, &nodeConfig, &device, &execution, &synchronizer, false);

    execute(&executor, &execution, &device);
}

void testRmsNorm_F32_F32_F32() {
    #define RMS_NORM_DIM 256
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, RMS_NORM_DIM));
            NnUint invRmsBufferIndex = nodeBuilder->addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1));
            segmentBuilder->addOp(OP_INV_RMS, "inv_rms", 0,
                pointerConfig(PNTR_PIPE, xPipeIndex),
                pointerConfig(PNTR_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{1e-5f});
            segmentBuilder->addOp(OP_RMS_NORM, "rms_norm", 0,
                pointerConfig(PNTR_PIPE, xPipeIndex),
                pointerConfig(PNTR_PIPE, xPipeIndex),
                size1D(F_32, RMS_NORM_DIM),
                NnRmsNormOpConfig{invRmsBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            std::vector<float> normWeight(RMS_NORM_DIM);
            for (NnUint i = 0; i < RMS_NORM_DIM; i++)
                normWeight[i] = (0.25f + (float)i) / (float)RMS_NORM_DIM;
            executor->loadWeight("rms_norm", 0, normWeight.size() * sizeof(float), (NnByte *)normWeight.data());

            float *xPipe = (float *)execution->pipes[0];
            for (NnUint i = 0; i < RMS_NORM_DIM; i++)
                xPipe[i] = (float)(RMS_NORM_DIM - i) / (float)(RMS_NORM_DIM / 2);

            // act
            executor->forward();

            // assert
            float invRmsBuffer[N_BATCHES];
            device->data->buffers[0].get()->read((NnByte *)invRmsBuffer);

            float t = 0.000001f;
            assertFloat(invRmsBuffer[0], 0.863493f, t);
            assertFloat(xPipe[0], 0.001687f, t);
            assertFloat(xPipe[1], 0.008400f, t);
            assertFloat(xPipe[2], 0.015060f, t);
            assertFloat(xPipe[35], 0.205286f, t);
            assertFloat(xPipe[36], 0.210155f, t);
            assertFloat(xPipe[119], 0.430514f, t);
            assertFloat(xPipe[123], 0.431964f, t);
            assertFloat(xPipe[234], 0.135804f, t);
            assertFloat(xPipe[242], 0.089372f, t);
            assertFloat(xPipe[249], 0.045977f, t);
            assertFloat(xPipe[255], 0.006726f, t);

            printOk("testRmsNorm_F32_F32_F32");
        });
}

int main() {
    testRmsNorm_F32_F32_F32();
    return 0;
}
