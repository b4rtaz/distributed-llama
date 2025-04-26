#include "nn-core.hpp"
#include "nn-config-builder.hpp"
#include "nn-cpu.hpp"
#include <cstdio>

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
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{1e-5f});

    segmentBuilder.addOp(OP_RMS_NORM, "rms_norm", 0,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        size1D(F_32, DIM),
        NnRmsNormOpConfig{invRmsBufferIndex});

    nodeBuilder.addSegment(segmentBuilder.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
}

void print2D(const char *name, NnUint x, NnUint y, float *w) {
    for (NnUint i = 0; i < y; i++) {
        printf("%s[%d] = ", name, i);
        for (NnUint j = 0; j < x; j++)
            printf("%f ", w[i * x + j]);
        printf("\n");
    }
}

int main() {
    initQuants();

    NnUint nThreads = 2;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    buildConfig(&netConfig, &nodeConfig);

    NnNetExecution execution(nThreads, &netConfig);
    float *x = (float *)execution.pipes[0];
    for (NnUint b = 0; b < N_BATCHES; b++) {
        for (NnUint i = 0; i < DIM; i++)
            x[b * DIM + i] = i / (float)DIM + (float)b;
    }

    print2D("x", DIM, N_BATCHES, x);

    float rmsNormWeight[DIM];
    for (NnUint i = 0; i < DIM; i++)
        rmsNormWeight[i] = 0.5 + i / (float)DIM;

    NnCpuDevice *device = new NnCpuDevice(&netConfig, &nodeConfig, &execution);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(device, -1, -1));

    NnFakeNodeSynchronizer synchronizer;
    float *rms = (float *)device->buffers[0];
    NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);
    executor.loadWeight("rms_norm", 0, sizeof(rmsNormWeight), (NnByte *)rmsNormWeight);

    execution.setBatchSize(2);
    executor.forward();

    print2D("rms", N_BATCHES, 1, rms);
    print2D("x", DIM, N_BATCHES, x);

    releaseNetConfig(&netConfig);
    releaseNodeConfig(&nodeConfig);
    return 0;
}