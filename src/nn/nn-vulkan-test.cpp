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
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{1e-5f});
            segmentBuilder->addOp(OP_RMS_NORM, "rms_norm", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size1D(F_32, RMS_NORM_DIM),
                NnRmsNormOpConfig{invRmsBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            const NnUint batchSize = 2;
            execution->setBatchSize(batchSize);

            std::vector<float> normWeight(RMS_NORM_DIM);
            for (NnUint i = 0; i < RMS_NORM_DIM; i++)
                normWeight[i] = (0.25f + (float)i) / (float)RMS_NORM_DIM;
            executor->loadWeight("rms_norm", 0, normWeight.size() * sizeof(float), (NnByte *)normWeight.data());

            float *xPipe = (float *)execution->pipes[0];
            for (NnUint b = 0; b < batchSize; b++) {
                float *xBatchPipe = &xPipe[b * RMS_NORM_DIM];
                for (NnUint i = 0; i < RMS_NORM_DIM; i++)
                    xBatchPipe[i] = (float)(RMS_NORM_DIM - i) / (float)(RMS_NORM_DIM / 2);
            }

            // act
            executor->forward();

            // assert
            float invRmsBuffer[N_BATCHES];
            device->data->buffers[0].get()->read((NnByte *)invRmsBuffer);

            for (NnUint b = 0; b < batchSize; b++) {
                float *xBatchPipe = &xPipe[b * RMS_NORM_DIM];

                float t = 0.000001f;
                assertFloat(invRmsBuffer[b], 0.863493f, t);
                assertFloat(xBatchPipe[0], 0.001687f, t);
                assertFloat(xBatchPipe[1], 0.008400f, t);
                assertFloat(xBatchPipe[2], 0.015060f, t);
                assertFloat(xBatchPipe[35], 0.205286f, t);
                assertFloat(xBatchPipe[36], 0.210155f, t);
                assertFloat(xBatchPipe[119], 0.430514f, t);
                assertFloat(xBatchPipe[123], 0.431964f, t);
                assertFloat(xBatchPipe[234], 0.135804f, t);
                assertFloat(xBatchPipe[242], 0.089372f, t);
                assertFloat(xBatchPipe[249], 0.045977f, t);
                assertFloat(xBatchPipe[255], 0.006726f, t);
            }
            printOk("testRmsNorm_F32_F32_F32");
        });
}


void testSilu_F32_F32() {
    #define SILU_DIM 32
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, SILU_DIM));
            segmentBuilder->addOp(OP_SILU, "silu", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnSiluOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(1);

            float *xPipe = (float *)execution->pipes[0];
            for (NnUint i = 0; i < SILU_DIM; i++)
                xPipe[i] = i / (float)SILU_DIM;

            // act
            executor->forward();

            // assert
            float t = 0.0006f;
            assertFloat(xPipe[0], 0.0f, t);
            assertFloat(xPipe[2], 0.032226f, t);
            assertFloat(xPipe[6], 0.102513f, t);
            assertFloat(xPipe[17], 0.334573f, t);
            assertFloat(xPipe[28], 0.617802f, t);
            assertFloat(xPipe[31], 0.702729f, t);

            printOk("testSilu_F32_F32");
        });
}

void testMul_F32_F32() {
    #define MUL_DIM 32
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MUL_DIM));
            NnUint sBufferIndex = nodeBuilder->addBuffer("s", size2D(F_32, N_BATCHES, MUL_DIM));
            segmentBuilder->addOp(OP_MUL, "mul", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMulOpCodeConfig{sBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];
            float sBuffer[MUL_DIM * N_BATCHES];
            for (NnUint i = 0; i < MUL_DIM * N_BATCHES; i++) {
                xPipe[i] = (float)i;
                sBuffer[i] = cosf((float)i);
            }

            device->data->buffers[0].get()->write((NnByte *)sBuffer);

            // act
            executor->forward();

            // assert
            for (NnUint i = 0; i < MUL_DIM * N_BATCHES; i++)
                assertFloat(xPipe[i], i * cosf((float)i), 0.00001f);
            printOk("testMul_F32_F32");
        });
}

int main() {
    testRmsNorm_F32_F32_F32();
    testSilu_F32_F32();
    testMul_F32_F32();
    return 0;
}
