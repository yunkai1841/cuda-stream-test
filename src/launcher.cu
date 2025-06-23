#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include "matrixmul.cuh"

// カーネルタイプに基づいて適切なカーネルを実行
void launchMatrixMulKernel(const char* kernel_type, float* d_C, const float* d_A, const float* d_B, int N, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    if (strcmp(kernel_type, "basic") == 0) {
        matrixMulKernel<<<gridSize, blockSize, 0, stream>>>(d_C, d_A, d_B, N);
    } else if (strcmp(kernel_type, "tiling") == 0) {
        matrixMulTilingKernel<<<gridSize, blockSize, 0, stream>>>(d_C, d_A, d_B, N);
    } else if (strcmp(kernel_type, "shared") == 0) {
        matrixMulTilingSharedKernel<<<gridSize, blockSize, 0, stream>>>(d_C, d_A, d_B, N);
    } else if (strcmp(kernel_type, "unroll") == 0) {
        matrixMulTilingSharedUnrollKernel<<<gridSize, blockSize, 0, stream>>>(d_C, d_A, d_B, N);
    } else {
        std::cerr << "Unknown kernel type: " << kernel_type << std::endl;
        std::cerr << "Available types: basic, tiling, shared, unroll" << std::endl;
        exit(1);
    }
}
