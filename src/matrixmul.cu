/**
 * Matrix multiplication using CUDA (tiled, boundary-safe)
 * C = A x B, where A, B, C are N x N row-major matrices.
 * Tile size is 16x16. Works for any N (not only multiples of 16).
 */
__global__ void matrixMulKernel(float *C, const float *A, const float *B, int N) {
    constexpr int TILE = 16;

    // Global row/col this thread computes
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Shared tiles
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;

    // Number of tiles along the K dimension
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;  // k-index for A
        int bRow = t * TILE + threadIdx.y;  // k-index for B

        // Load A tile if in bounds, else 0
        if (row < N && aCol < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile if in bounds, else 0
        if (bRow < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply partial tiles
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result if in bounds
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
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
