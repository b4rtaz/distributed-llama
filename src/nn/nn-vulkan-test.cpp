#include <cstdio>
#include "nn-config-builder.hpp"
#include "nn-vulkan.hpp"

#define N_BATCHES 4

void printOk(const char *name) {
    printf("✅ %24s passed\n", name);
}

void assertFloat(NnUint position, const float value, const float expectedValue, const float tolerance) {
    float diff = fabs(expectedValue - value);
    if (diff > tolerance) {
        printf("❌ [%d] failed: value=%f, expectedValue=%f, diff=%f\n", position, value, expectedValue, diff);
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
                assertFloat(b, invRmsBuffer[b], 0.863493f, t);
                assertFloat(0, xBatchPipe[0], 0.001687f, t);
                assertFloat(1, xBatchPipe[1], 0.008400f, t);
                assertFloat(2, xBatchPipe[2], 0.015060f, t);
                assertFloat(35, xBatchPipe[35], 0.205286f, t);
                assertFloat(36, xBatchPipe[36], 0.210155f, t);
                assertFloat(119, xBatchPipe[119], 0.430514f, t);
                assertFloat(123, xBatchPipe[123], 0.431964f, t);
                assertFloat(234, xBatchPipe[234], 0.135804f, t);
                assertFloat(242, xBatchPipe[242], 0.089372f, t);
                assertFloat(249, xBatchPipe[249], 0.045977f, t);
                assertFloat(255, xBatchPipe[255], 0.006726f, t);
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
            assertFloat(0, xPipe[0], 0.0f, t);
            assertFloat(2, xPipe[2], 0.032226f, t);
            assertFloat(6, xPipe[6], 0.102513f, t);
            assertFloat(17, xPipe[17], 0.334573f, t);
            assertFloat(28, xPipe[28], 0.617802f, t);
            assertFloat(31, xPipe[31], 0.702729f, t);

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
                assertFloat(i, xPipe[i], i * cosf((float)i), 0.00001f);
            printOk("testMul_F32_F32");
        });
}

void testMergeAdd_F32_F32() {
    #define MERGE_ADD_NODES 2
    #define MERGE_ADD_DIM 64
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint zPipeIndex = netBuilder->addPipe("Z", size2D(F_32, N_BATCHES, MERGE_ADD_DIM * MERGE_ADD_NODES));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MERGE_ADD_DIM));
            segmentBuilder->addOp(OP_MERGE_ADD, "mergeAdd", 0,
                pointerBatchConfig(SRC_PIPE, zPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float *zPipe = (float *)execution->pipes[0];
            float *xPipe = (float *)execution->pipes[1];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint n = 0; n < MERGE_ADD_NODES; n++) {
                    for (NnUint i = 0; i < MERGE_ADD_DIM; i++)
                        zPipe[b * MERGE_ADD_NODES * MERGE_ADD_DIM + n * MERGE_ADD_DIM + i] = (float)(b + 1);
                }
            }

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < MERGE_ADD_DIM; i++) {
                    NnUint j = b * MERGE_ADD_DIM + i;
                    assertFloat(j, xPipe[j], (float)(2 * b + 2), 0.00001f);
                }
            }
            printOk("testMergeAdd_F32_F32");
        });
}

void testEmbedding_F32_F32() {
    #define EMBEDDING_DIM 16
    #define EMBEDDING_LEN 8
    assert(EMBEDDING_LEN > N_BATCHES);
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, EMBEDDING_DIM));
            segmentBuilder->addOp(OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, posPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size2D(F_32, EMBEDDING_LEN, EMBEDDING_DIM),
                NnEmbeddingOpConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float embedding[EMBEDDING_DIM * EMBEDDING_LEN];
            for (NnUint l = 0; l < EMBEDDING_LEN; l++) {
                for (NnUint i = 0; i < EMBEDDING_DIM; i++)
                    embedding[l * EMBEDDING_DIM + i] = (float)(l + 4);
            }
            float *posPipe = (float *)execution->pipes[0];
            for (NnUint b = 0; b < N_BATCHES; b++)
                posPipe[b] = (float)b;

            executor->loadWeight("embedding", 0, EMBEDDING_DIM * EMBEDDING_LEN * sizeof(float), (NnByte *)embedding);

            // act
            executor->forward();

            // assert
            float *xPipe = (float *)execution->pipes[1];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < EMBEDDING_DIM; i++) {
                    NnUint j = b * EMBEDDING_DIM + i;
                    assertFloat(j, xPipe[j], (float)(b + 4), 0.00001f);
                }
            }

            printOk("testEmbedding_F32_F32");
        });
}

void testShift_F32_F32() {
    #define SHIFT_DIM 48
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, SHIFT_DIM));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, 1, N_BATCHES * SHIFT_DIM));
            segmentBuilder->addOp(
                OP_SHIFT, "shift", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerRawConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnShiftOpCodeConfig{posPipeIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[1];
            float *yPipe = (float *)execution->pipes[2];

            float pos[N_BATCHES];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                pos[b] = (float)b;
                for (NnUint i = 0; i < SHIFT_DIM; i++)
                    xPipe[b * SHIFT_DIM + i] = (float)(b * 100 + i);
            }

            device->data->pipes[0].get()->write((NnByte *)pos);

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < SHIFT_DIM; i++) {
                    NnUint j = b * SHIFT_DIM + i;
                    assertFloat(j, yPipe[j], (float)(b * 100 + i), 0.00001f);
                }
            }
            printOk("testShift_F32_F32");
        });
}

int main() {
    testRmsNorm_F32_F32_F32();
    testSilu_F32_F32();
    testMul_F32_F32();
    testMergeAdd_F32_F32();
    testEmbedding_F32_F32();
    testShift_F32_F32();
    return 0;
}
