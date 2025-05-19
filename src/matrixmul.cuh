__global__ void matrixMulKernel(float *C, const float *A, const float *B, int N);
__global__ void matrixMulTilingKernel(float *C, const float *A, const float *B, int N);
__global__ void matrixMulTilingSharedKernel(float *C, const float *A, const float *B, int N);
__global__ void matrixMulTilingSharedUnrollKernel(float *C, const float *A, const float *B, int N);

