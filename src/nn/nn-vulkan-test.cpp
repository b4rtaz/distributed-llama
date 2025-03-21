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

    NnUint gpuIndex = 0;
    NnVulkanDevice device(gpuIndex, &netConfig, &nodeConfig, &execution);
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
                for (NnUint i = 0; i < RMS_NORM_DIM; i++) {
                    float u = (float)(RMS_NORM_DIM - i + b) / (float)(RMS_NORM_DIM / 2);
                    xBatchPipe[i] = u;
                }
            }

            // act
            executor->forward();

            // assert
            float invRmsBuffer[N_BATCHES];
            device->data->buffers[0].get()->read((NnByte *)invRmsBuffer);

            float expectedS[N_BATCHES];
            expectedS[0] = 0.863493f;
            expectedS[1] = 0.858468f;

            for (NnUint b = 0; b < batchSize; b++) {
                float *xBatchPipe = &xPipe[b * RMS_NORM_DIM];

                const float t = 0.000001f;
                const float s = expectedS[b];
                assertFloat(b, invRmsBuffer[b], s, t);
                for (NnUint i = 0; i < RMS_NORM_DIM; i++) {
                    float u = (float)(RMS_NORM_DIM - i + b) / (float)(RMS_NORM_DIM / 2);
                    assertFloat(b * RMS_NORM_DIM + i, xBatchPipe[i], (u * s) * normWeight[i], t);
                }
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
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                float *x = &xPipe[b * SILU_DIM];
                for (NnUint i = 0; i < SILU_DIM; i++)
                    x[i] = i / (float)SILU_DIM;
            }

            // act
            executor->forward();

            // assert
            float t = 0.0006f;
            for (NnUint b = 0; b < N_BATCHES; b++) {
                float *x = &xPipe[b * SILU_DIM];
                assertFloat(0, x[0], 0.0f, t);
                assertFloat(2, x[2], 0.032226f, t);
                assertFloat(6, x[6], 0.102513f, t);
                assertFloat(17, x[17], 0.334573f, t);
                assertFloat(28, x[28], 0.617802f, t);
                assertFloat(31, x[31], 0.702729f, t);
            }

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
                sBuffer[i] = (i % 8) / 10.0f;
            }

            device->data->buffers[0].get()->write((NnByte *)sBuffer);

            // act
            executor->forward();

            // assert
            for (NnUint i = 0; i < MUL_DIM * N_BATCHES; i++)
                assertFloat(i, xPipe[i], i * ((i % 8) / 10.0f), 0.000001f);
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

void testCast_F32_F32() {
    #define CAST_DIM 64
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, CAST_DIM));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, N_BATCHES, CAST_DIM));
            segmentBuilder->addOp(
                OP_CAST, "cast", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            float *yPipe = (float *)execution->pipes[1];

            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < CAST_DIM; i++)
                    xPipe[b * CAST_DIM + i] = (float)b;
            }

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < CAST_DIM; i++) {
                    NnUint j = b * CAST_DIM + i;
                    assertFloat(j, yPipe[j], (float)b, 0.00001f);
                }
            }
            printOk("testCast_F32_F32");
        });
}

void testRope_F32_F32() {
    #define ROPE_DIM 2048
    #define ROPE_KV_DIM 512
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint nHeads = 32;
            const NnUint seqLen = 4096;
            const NnRopeSlice slice = sliceRope(ROPE_DIM, ROPE_KV_DIM, 8, 1, seqLen, ROPE_DIM / nHeads, 500000.0f, 0);

            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, ROPE_DIM));
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint ropeCacheBufferIndex = nodeBuilder->addBuffer("ropeCache", slice.cacheSize);
            bool isQ = true;

            segmentBuilder->addOp(
                OP_ROPE_LLAMA, "rope_llama", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnRopeLlamaOpConfig{isQ, posPipeIndex, ropeCacheBufferIndex, 32.0f, 1.0f, 4.0f, 8192, slice});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(2);

            float *xPipe = (float *)execution->pipes[0];
            float pos[N_BATCHES];
            pos[0] = (float)6;
            pos[1] = (float)31;

            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < ROPE_DIM; i++)
                    xPipe[b * ROPE_DIM + i] = 1.0f;
            }

            device->data->pipes[1].get()->write((NnByte *)pos);

            // act
            executor->forward();

            // assert
            float t = 0.000001f;

            float *x0 = &xPipe[0 * ROPE_DIM];
            assertFloat(0, x0[0], 1.239586f, t);
            assertFloat(1, x0[1], 0.680755f, t);
            assertFloat(2, x0[2], 0.077202f, t);
            assertFloat(3, x0[3], -1.412105f, t);
            assertFloat(1988, x0[1988], -1.356766f, t);
            assertFloat(2022, x0[2022], 0.999923, t);
            assertFloat(2023, x0[2023], 1.000077, t);

            float *x1 = &xPipe[1 * ROPE_DIM];
            assertFloat(0, x1[0], 1.318780f, t);
            assertFloat(1, x1[1], 0.510705f, t);
            assertFloat(1078, x1[1078], 0.999985f, t);
            assertFloat(1078, x1[1079], 1.000015f, t);

            printOk("testRope_F32_F32");
        });
}

void matmul_F32_F32_F32() {
    #define MATMUL_N 64
    #define MATMUL_D 96
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MATMUL_N));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, N_BATCHES, MATMUL_D));
            segmentBuilder->addOp(
                OP_MATMUL, "matmul", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size2D(F_32, MATMUL_N, MATMUL_D),
                NnMatmulOpConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            float *yPipe = (float *)execution->pipes[1];

            float weight[MATMUL_N * MATMUL_D];
            for (NnUint i = 0; i < N_BATCHES * MATMUL_N; i++)
                xPipe[i] = i * 0.01f;
            for (NnUint i = 0; i < MATMUL_N * MATMUL_D; i++)
                weight[i] = i * 0.001f;
            executor->loadWeight("matmul", 0, MATMUL_N * MATMUL_D * sizeof(float), (NnByte *)weight);

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint d = 0; d < MATMUL_D; d++) {
                    float sum = 0.0f;
                    for (NnUint n = 0; n < MATMUL_N; n++)
                        sum += xPipe[b * MATMUL_N + n] * weight[d * MATMUL_N + n];

                    const NnUint p = b * MATMUL_D + d;
                    assertFloat(p, yPipe[p], sum, 0.0002f);
                }
            }
            printOk("matmul_F32_F32_F32");
        });
}

void multiheadAtt_F32_F32() {
    #define MULTIHEAD_ATT_DIM 128
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint nHeads = 32;
            const NnUint nKvHeads = 8;
            const NnUint headSize = MULTIHEAD_ATT_DIM / nHeads;
            const NnUint seqLen = 4096;
            const NnUint qSliceD0 = 2048;
            const NnUint kvDim0 = 512;
            const NnKvCacheSlice kvCacheSlice = sliceKvCache(kvDim0, seqLen, 1);
            const NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(nHeads, seqLen, 1);

            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MULTIHEAD_ATT_DIM));
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint qBufferIndex = nodeBuilder->addBuffer("POS", size2D(F_32, N_BATCHES, qSliceD0));
            NnUint kCacheBufferIndex = nodeBuilder->addBuffer("kCache", kvCacheSlice.keySize);
            NnUint vCacheBufferIndex = nodeBuilder->addBuffer("vCache", kvCacheSlice.valueSize);
            NnUint attCacheBufferIndex = nodeBuilder->addBuffer("vCache", multiHeadAttSlice.attSize);

            segmentBuilder->addOp(
                OP_MULTIHEAD_ATT, "multihead_att", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMultiHeadAttOpConfig{nHeads, nHeads, nKvHeads, headSize, seqLen, qSliceD0, kvDim0,
                    posPipeIndex, qBufferIndex, kCacheBufferIndex, vCacheBufferIndex, attCacheBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // TODO: for now this is a smoke test
            execution->setBatchSize(N_BATCHES);
            executor->forward();
            printOk("multiheadAtt_F32_F32");
        });
}

int main() {
    testRmsNorm_F32_F32_F32();
    testSilu_F32_F32();
    testMul_F32_F32();
    testMergeAdd_F32_F32();
    testEmbedding_F32_F32();
    testShift_F32_F32();
    testCast_F32_F32();
    testRope_F32_F32();
    matmul_F32_F32_F32();
    multiheadAtt_F32_F32();
    return 0;
}
