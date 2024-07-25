#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cassert>
#include "utils.hpp"
#include "funcs.hpp"
#include "transformer.hpp"
#include "tasks.hpp"
#include "llama2-tasks.hpp"
#include "grok1-tasks.hpp"

float expectedOutput_0_4[] = { 0.00940248929, 0.0191232786, 0.0147766126, 0.0102868658 };
float expectedOutput_256_260[] = { 0.0191071425, 0.0134582901, 0.0146755828, 0.019181719 };
float expectedOutput_5012_5016[] = { 0.0126675405, 0.0169415697, 0.0183475353, 0.0182626117 };

void compare(float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        if (std::isnan(a[i]) || fabs(a[i] - b[i]) > 0.000035) { // Optimization may cause some differences
            printf("%.9g != %.9g\n", a[i], b[i]); i++;
            printf("%.9g != %.9g\n", a[i], b[i]); i++;
            printf("%.9g != %.9g\n", a[i], b[i]); i++;
            printf("%.9g != %.9g\n", a[i], b[i]); i++;
            exit(EXIT_FAILURE);
        }
    }
}

int main() {
    TransformerSpec spec;
    spec.headerSize = sizeof(TransformerFileOldHeader) + sizeof(int);
    spec.archType = GROK1;
    spec.ropeType = ROPE_FALCON;
    spec.dim = 6144;
    spec.nLayers = 1;
    spec.nHeads = 48;
    spec.headSize = spec.dim / spec.nHeads;
    spec.nKvHeads = 8;
    spec.seqLen = 8192;
    spec.hiddenDim = 1024;
    spec.kvDim = (spec.dim * spec.nKvHeads) / spec.nHeads;
    spec.vocabSize = 1024;
    spec.nExperts = 8;
    spec.nActiveExperts = 2;
    spec.weightsFloatType = F32;
    spec.bufferFloatType = F32;
    spec.nSlices = 1;
    spec.hiddenAct = GELU;
    spec.ropeTheta = 10000.0f;

    TransformerConfig config;
    config.useDiscForKvCache = false;

    size_t beforeBlockBytes = spec.dim * spec.vocabSize * sizeof(float);
    size_t blockBytes = 956596224;
    size_t afterBlockBytes  = (spec.dim + spec.dim * spec.vocabSize) * sizeof(float);
    spec.fileSize = spec.headerSize + beforeBlockBytes + blockBytes + afterBlockBytes;

    char* weights = (char*)newBuffer(beforeBlockBytes + blockBytes + afterBlockBytes);
    long nFloats = blockBytes / sizeof(float);
    float* block = (float*)&weights[beforeBlockBytes];

    unsigned long long state = 123456789L;
    for (int f = 0; f < nFloats; f++) block[f] = randomF32(&state) / 100.0;

    SocketPool socketPool(0, NULL);
    Transformer transformer = Transformer::loadRoot(weights, &spec, &config, &socketPool);
    transformer.pos = 0;

    float* x = transformer.x;
    for (int i = 0; i < spec.dim; i++) x[i] = (randomF32(&state) / 100.0) / 78.38367176906169f;

    TransformerArch arch = buildGrok1Arch(&spec);

    int nThreads = 4;
    TransformerContext context;
    context.transformer = &transformer;
    context.currentBlockIndex = 0;
    context.socket = NULL;
    context.socketPool = &socketPool;

    int skipLastNTasks = 4;
    TaskLoop loop(nThreads, arch.inference.nTasks - skipLastNTasks, TASK_N_TYPES, arch.inference.tasks, &context);
    long t0 = timeMs();
    loop.run();
    long t1 = timeMs();

    freeBuffer(weights);

    compare(&x[0], expectedOutput_0_4, 4);
    compare(&x[256], expectedOutput_256_260, 4);
    compare(&x[5012], expectedOutput_5012_5016, 4);

    printf("✅ Block forwarded correctly in %ldms\n", t1 - t0);
}
