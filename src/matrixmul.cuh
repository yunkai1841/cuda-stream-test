__global__ void matrixMulKernel(float *C, const float *A, const float *B, int N);
__global__ void matrixMulTilingKernel(float *C, const float *A, const float *B, int N);
__global__ void matrixMulTilingSharedKernel(float *C, const float *A, const float *B, int N);
__global__ void matrixMulTilingSharedUnrollKernel(float *C, const float *A, const float *B, int N);
__global__ void vectorAddKernel(float *C, const float *A, const float *B, int N);

// CUDA kernel launcher function
void launchMatrixMulKernel(const char* kernel_type, float* d_C, const float* d_A, const float* d_B, int N, cudaStream_t stream = 0);

