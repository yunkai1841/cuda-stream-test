#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <time.h>

#define N 1024  // Matrix size N x N

__global__ void matrixMulKernel(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void randomInit(float *data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = (float)(rand() % 100) / 10.0f;
}

int main(int argc, char **argv) {
    int size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize random seeds
    srand(time(NULL));
    randomInit(h_A, N * N);
    randomInit(h_B, N * N);

    cudaStream_t streams[10];
    for (int i = 0; i < 10; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

    float *d_A, *d_B, *d_C;
    float *clone_A[10], *clone_B[10], *clone_C[10];
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create 10 clones of A, B, and C
    for (int i = 0; i < 10; ++i) {
        cudaMallocAsync((void **)&clone_A[i], size, streams[i]);
        cudaMallocAsync((void **)&clone_B[i], size, streams[i]);
        cudaMallocAsync((void **)&clone_C[i], size, streams[i]);
        cudaMemcpyAsync(clone_A[i], d_A, size, cudaMemcpyDeviceToDevice, streams[i]);
        cudaMemcpyAsync(clone_B[i], d_B, size, cudaMemcpyDeviceToDevice, streams[i]);
        cudaMemcpyAsync(clone_C[i], d_C, size, cudaMemcpyDeviceToDevice, streams[i]);
    }

    for (int i = 0; i < 10; ++i) {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch the kernel with the stream
        matrixMulKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(clone_A[i], clone_B[i], clone_C[i], N);
    }

    // dim3 threadsPerBlock(16, 16);
    // dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print a small part of the result for verification
    // printf("C[0][0] = %f\n", h_C[0]);
    // printf("C[N-1][N-1] = %f\n", h_C[N*N-1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    for (int i = 0; i < 10; ++i) {
        cudaFree(clone_A[i]);
        cudaFree(clone_B[i]);
        cudaFree(clone_C[i]);
        cudaStreamDestroy(streams[i]);
    }
    // cudaStreamDestroy(stream);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}