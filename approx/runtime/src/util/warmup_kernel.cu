#include "approx_debug.h"


__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

namespace approx { namespace util {
    void warmup() {
        warm_up_gpu<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
}
}

