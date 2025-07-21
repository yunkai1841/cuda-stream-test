#include "cuda_utils.h"
#include <cstdlib>  // for std::getenv, std::system
#include <sys/wait.h>  // for WEXITSTATUS
#include <iostream>  // for std::cout

namespace cuda_utils {

// CudaMemory implementation
CudaMemory::CudaMemory(size_t size) : ptr_(nullptr), size_(size) {
    cudaError_t err = cudaMalloc(&ptr_, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed: " +
                                 std::string(cudaGetErrorString(err)));
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
        throw std::runtime_error("CUDA stream creation failed: " +
                                 std::string(cudaGetErrorString(err)));
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

void CudaStream::synchronize() const { cudaStreamSynchronize(stream_); }

// CudaTimer implementation
CudaTimer::CudaTimer() : start_(nullptr), stop_(nullptr) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

CudaTimer::~CudaTimer() {
    if (start_) cudaEventDestroy(start_);
    if (stop_) cudaEventDestroy(stop_);
}

void CudaTimer::start() {
    cudaEventRecord(start_, 0);
}

void CudaTimer::stop() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
}

float CudaTimer::elapsedMilliseconds() const {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
}

// CudaMPS implementation
bool CudaMPS::isAvailable() {
    // CUDAデバイスのCompute Capabilityをチェック
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        return false;
    }

    int device;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return false;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return false;
    }

    // Compute Capability 3.5以上でMPSがサポート
    return (prop.major > 3 || (prop.major == 3 && prop.minor >= 5));
}

bool CudaMPS::isRunning() {
    // CUDA_MPS_PIPE_DIRECTORY環境変数の存在をチェック
    const char* mps_pipe = std::getenv("CUDA_MPS_PIPE_DIRECTORY");
    if (mps_pipe == nullptr) {
        // 従来のCUDA_MPS_PIPE変数もチェック
        mps_pipe = std::getenv("CUDA_MPS_PIPE");
    }
    
    if (mps_pipe == nullptr) {
        // デフォルトのパスもチェック
        mps_pipe = "/tmp/nvidia-mps";
    }

    // 実際のプロセス確認 (nvidia-cuda-mps-control)
    int result = std::system("pgrep -f nvidia-cuda-mps-control > /dev/null 2>&1");
    return (WEXITSTATUS(result) == 0);
}

std::string CudaMPS::getStatus() {
    std::string status = "MPS Status:\n";
    
    if (!isAvailable()) {
        status += "  - MPS not supported on this device\n";
        return status;
    }
    
    status += "  - MPS supported: Yes\n";
    
    if (isRunning()) {
        status += "  - MPS daemon running: Yes\n";
        
        // MPSの設定情報を取得
        const char* mps_pipe = std::getenv("CUDA_MPS_PIPE_DIRECTORY");
        if (mps_pipe) {
            status += "  - MPS pipe directory: " + std::string(mps_pipe) + "\n";
        }
        
        const char* active_thread_percentage = std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
        if (active_thread_percentage) {
            status += "  - Active thread percentage: " + std::string(active_thread_percentage) + "%\n";
        }
    } else {
        status += "  - MPS daemon running: No\n";
        status += "  - Use 'nvidia-cuda-mps-control -d' to start MPS\n";
    }
    
    return status;
}

void CudaMPS::printRecommendedSettings() {
    std::cout << "\nMPS Recommended Settings:\n";
    std::cout << "  To enable MPS:\n";
    std::cout << "    export CUDA_VISIBLE_DEVICES=0  # Set specific GPU\n";
    std::cout << "    nvidia-cuda-mps-control -d     # Start MPS daemon\n";
    std::cout << "\n  Optional environment variables:\n";
    std::cout << "    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50  # Limit GPU usage per client\n";
    std::cout << "    export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0x100000000  # Limit pinned memory\n";
    std::cout << "\n  To stop MPS:\n";
    std::cout << "    echo quit | nvidia-cuda-mps-control\n";
    std::cout << "\n";
}

}  // namespace cuda_utils
