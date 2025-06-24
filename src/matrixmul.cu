/**
 * Matrix multiplication using CUDA
 */
__global__ void matrixMulKernel(float *C, const float *A, const float *B, int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * 16 * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + N - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = 16;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = 16 * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = 16 * N;

    // Csub is used to store the element of C computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute Csub
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Shared memory for the sub-matrices of A and B
        __shared__ float As[16][16];
        __shared__ float Bs[16][16];

        // Load the matrices from global memory to shared memory
        As[ty][tx] = A[a + N * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together
        for (int k = 0; k < 16; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding computation is done before loading new data
        // in shared memory
        __syncthreads();
    }

    // Write Csub to global memory
    int cIndex = N * 16 * by + 16 * bx;
    C[cIndex + N * ty + tx] = Csub;
}

/**
 * Matrix multiplication using tiling approach
 */
__global__ void matrixMulTilingKernel(float *C, const float *A, const float *B, int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * 16 * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + N - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = 16;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = 16 * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = 16 * N;

    // Csub is used to store the element of C computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B required to compute Csub
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from global memory to shared memory
        Csub += A[a + N * ty + tx] * B[b + N * ty + tx];
    }

    // Write Csub to global memory
    int cIndex = N * 16 * by + 16 * bx;
    C[cIndex + N * ty + tx] = Csub;
}

/**
 * Matrix multiplication using tiling approach with shared memory
 */
__global__ void matrixMulTilingSharedKernel(float *C, const float *A, const float *B, int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * 16 * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + N - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = 16;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = 16 * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = 16 * N;

    // Csub is used to store the element of C computed by the thread
    float Csub = 0;

    // Shared memory for the sub-matrices of A and B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    // Loop over all the sub-matrices of A and B required to compute Csub
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from global memory to shared memory
        As[ty][tx] = A[a + N * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together
        for (int k = 0; k < 16; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding computation is done before loading new data
        // in shared memory
        __syncthreads();
    }

    // Write Csub to global memory
    int cIndex = N * 16 * by + 16 * bx;
    C[cIndex + N * ty + tx] = Csub;
}

/**
 * matrix multiplication using tiling approach with shared memory and unrolling
 */
__global__ void matrixMulTilingSharedUnrollKernel(float *C, const float *A, const float *B, int N) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = N * 16 * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + N - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = 16;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = 16 * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = 16 * N;

    // Csub is used to store the element of C computed by the thread
    float Csub = 0;

    // Shared memory for the sub-matrices of A and B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    // Loop over all the sub-matrices of A and B required to compute Csub
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from global memory to shared memory
        As[ty][tx] = A[a + N * ty + tx];
        Bs[ty][tx] = B[b + N * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

// Multiply the two matrices together with unrolling
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        // Synchronize to make sure that the preceding computation is done before loading new data
        // in shared memory
        __syncthreads();
    }
    // Write Csub to global memory
    int cIndex = N * 16 * by + 16 * bx;
    C[cIndex + N * ty + tx] = Csub;
}

/**
 * Vector addition using CUDA
 */
__global__ void vectorAddKernel(float *C, const float *A, const float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
