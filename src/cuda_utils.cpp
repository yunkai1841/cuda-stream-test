#include "cuda_utils.h"

namespace cuda_utils {

// CudaMemory implementation
CudaMemory::CudaMemory(size_t size) : ptr_(nullptr), size_(size) {
    cudaError_t err = cudaMalloc(&ptr_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
    }
}

CudaMemory::~CudaMemory() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
}

CudaMemory::CudaMemory(CudaMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

CudaMemory& CudaMemory::operator=(CudaMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_) cudaFree(ptr_);
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

float* CudaMemory::release() {
    float* temp = ptr_;
    ptr_ = nullptr;
    size_ = 0;
    return temp;
}

void CudaMemory::reset(float* new_ptr, size_t new_size) {
    if (ptr_) cudaFree(ptr_);
    ptr_ = new_ptr;
    size_ = new_size;
}

// CudaStream implementation
CudaStream::CudaStream() : stream_(nullptr) {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA stream creation failed: " + std::string(cudaGetErrorString(err)));
    }
}

CudaStream::~CudaStream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CudaStream::synchronize() const {
    cudaStreamSynchronize(stream_);
}

} // namespace cuda_utils
