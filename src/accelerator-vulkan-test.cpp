#include "utils.hpp"
#include "funcs.hpp"
#include "accelerator-vulkan.hpp"

bool compareOutputs(unsigned int d, float* cpu, float* gpu) {
    for (unsigned int i = 0; i < d; i++) {
        if (fabs(cpu[i] - gpu[i]) > 0.01) {
            for (unsigned int j = i; j < d; j++)
                printf("🚨 [%3d] cpu %.9g != gpu %.9g\n", j, cpu[j], gpu[j]);
            return false;
        }
    }
    return true;
}

int main() {
    initQuants();
    VulkanContext context = VulkanContext();
    AcceleratorVulkan accelerator = AcceleratorVulkan(&context);

    bool success = true;
    unsigned long long state = 1000000L;
    randomU32(&state);

    {
        const unsigned n = 512;
        const unsigned d = 128;
        float input[n];
        float weights[n * d];
        float outputGpu[d];
        float outputCpu[d];

        for (unsigned int i = 0; i < n * d; i++) {
            if (i < n) input[i] = (randomF32(&state) - 0.5f) / 2.0f;
            weights[i] = (randomF32(&state) - 0.5f) / 2.0f;
        }

        unsigned int mmIndex = accelerator.allocateMatmul(F32, F32, n, d);
        accelerator.loadMatmulWeights(mmIndex, weights);
        accelerator.beginForwardMatmul(mmIndex, input);
        accelerator.endForwardMatmul(mmIndex, outputGpu);

        matmul(F32, F32, outputCpu, input, weights, n, d, 1, 0);

        DEBUG_FLOATS("matmul f32xf32", outputGpu, 8);

        if (compareOutputs(d, outputCpu, outputGpu))
            printf("✅ matmul f32xf32\n");
        else
            success = false;
    }

    for (unsigned n = 512; n <= 704; n += 32) {
        const unsigned d = 128;
        float input[n];
        const unsigned weightsN = (n * d) / QK40;
        BlockQ40 weights[weightsN];
        float outputGpu[d];
        float outputCpu[d];

        for (unsigned int i = 0; i < n; i++)
            input[i] = randomF32(&state) - 0.5f;
        for (unsigned int i = 0; i < weightsN; i++) {
            weights[i].d = 40000 + (int)(randomF32(&state) * 1000);
            for (unsigned int j = 0; j < 16; j++)
                weights[i].qs[j] = (int8_t)(randomF32(&state) * 256.0);
        }

        unsigned int mmIndex = accelerator.allocateMatmul(Q40, F32, n, d);

        accelerator.loadMatmulWeights(mmIndex, weights);
        accelerator.beginForwardMatmul(mmIndex, input);
        accelerator.endForwardMatmul(mmIndex, outputGpu);

        matmul(Q40, F32, outputCpu, input, weights, n, d, 1, 0);

        DEBUG_FLOATS("matmul q40xf32", outputGpu, 8);

        if (compareOutputs(d, outputCpu, outputGpu))
            printf("✅ matmul q40xf32 (n=%d)\n", n);
        else
            success = false;
    }

    for (unsigned n = 512; n <= 704; n += 32) {
        const unsigned d = 128;
        const unsigned inputN = n / QK80;
        const unsigned weightsN = (n * d) / QK40;
        BlockQ80 input[inputN];
        BlockQ40 weights[weightsN];
        float outputGpu[d];
        float outputCpu[d];

        for (unsigned int i = 0; i < inputN; i++) {
            input[i].d = 38000 + (int)(randomF32(&state) * 1000);
            for (unsigned int j = 0; j < 32; j++)
                input[i].qs[j] = (int8_t)(randomF32(&state) * 256.0);
        }
        for (unsigned int i = 0; i < weightsN; i++) {
            weights[i].d = 38000 + (int)(randomF32(&state) * 1000);
            for (unsigned int j = 0; j < 16; j++)
                weights[i].qs[j] = (int8_t)(randomF32(&state) * 256.0);
        }

        unsigned int mmIndex = accelerator.allocateMatmul(Q40, Q80, n, d);

        accelerator.loadMatmulWeights(mmIndex, weights);
        accelerator.beginForwardMatmul(mmIndex, input);
        accelerator.endForwardMatmul(mmIndex, outputGpu);

        DEBUG_FLOATS("matmul q40xq80", outputGpu, 8);

        matmul(Q40, Q80, outputCpu, input, weights, n, d, 1, 0);

        if (compareOutputs(d, outputCpu, outputGpu))
            printf("✅ matmul q40xq80 (n=%d)\n", n);
        else
            success = false;
    }

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
