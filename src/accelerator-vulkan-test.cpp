#include "accelerator-vulkan.hpp"

int main() {
    AcceleratorVulkan accelerator = AcceleratorVulkan();

    unsigned int mmIndex1 = accelerator.allocateMatmul(Q80, Q40, 512, 256);
    unsigned int mmIndex2 = accelerator.allocateMatmul(Q80, Q40, 512, 256);

    return 0;
}
