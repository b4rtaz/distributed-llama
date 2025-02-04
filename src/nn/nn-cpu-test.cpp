#include "nn-core.hpp"
#include "nn-config-builder.hpp"
#include "nn-cpu.hpp"

#define DIM 32
#define N_BATCHES 2

void buildConfig(NnNetConfig *netConfig, NnNodeConfig *nodeConfig) {
    NnSize nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, N_BATCHES);
    NnSize xPipeIndex = netBuilder.addPipe("X", size2D(F_32, N_BATCHES, DIM));

    NnNodeConfigBuilder nodeBuilder(0);
    NnSize rmsBufferIndex = nodeBuilder.addBuffer("rms", size2D(F_32, N_BATCHES, 1));
    NnSegmentConfigBuilder segmentBuilder;
    segmentBuilder.addSync(xPipeIndex, SYNC_NODE_SLICES_EXCEPT_ROOT);

    segmentBuilder.addOp(OP_RMS, "rms", 0,
        pointerConfig(PNTR_PIPE, xPipeIndex),
        pointerConfig(PNTR_BUFFER, rmsBufferIndex),
        size0(),
        NnRmsOpConfig{1e-5f});

    segmentBuilder.addOp(OP_RMS_NORM, "rmsNorm", 0,
        pointerConfig(PNTR_PIPE, xPipeIndex),
        pointerConfig(PNTR_PIPE, xPipeIndex),
        size1D(F_32, DIM),
        NnRmsNormOpConfig{rmsBufferIndex});

    nodeBuilder.addSegment(segmentBuilder.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
}

void print2D(const char *name, NnSize x, NnSize y, float *w) {
    for (NnSize i = 0; i < y; i++) {
        printf("%s[%d] = ", name, i);
        for (NnSize j = 0; j < x; j++)
            printf("%f ", w[i * x + j]);
        printf("\n");
    }
}

int main() {
    NnSize nThreads = 2;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    buildConfig(&netConfig, &nodeConfig);

    NnNetExecution execution(nThreads, &netConfig);
    float *x = (float *)execution.pipes[0];
    for (NnSize b = 0; b < N_BATCHES; b++) {
        for (NnSize i = 0; i < DIM; i++)
            x[b * DIM + i] = i / (float)DIM + (float)b;
    }

    print2D("x", DIM, N_BATCHES, x);

    float rmsNormWeight[DIM];
    for (NnSize i = 0; i < DIM; i++)
        rmsNormWeight[i] = 0.5 + i / (float)DIM;

    NnCpuDevice device(&netConfig, &nodeConfig, &execution);
    NnFakeNodeSynchronizer synchronizer;
    float *rms = (float *)device.buffers[0];
    NnExecutor executor(&netConfig, &nodeConfig, &device, &execution, &synchronizer);
    executor.loadWeight("rmsNorm", 0, sizeof(rmsNormWeight), (NnByte *)rmsNormWeight);

    execution.setBatchSize(2);
    executor.forward();

    print2D("rms", N_BATCHES, 1, rms);
    print2D("x", DIM, N_BATCHES, x);

    releaseNetConfig(&netConfig);
    releaseNodeConfig(&nodeConfig);
    return 0;
}