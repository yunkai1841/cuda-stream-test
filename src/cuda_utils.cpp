#include "cuda_utils.h"

#include <cstdlib>  // for system, setenv, getenv
#include <iostream> // for std::cerr
#include <cstdio>   // for popen, pclose

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

// CudaMps implementation
bool CudaMps::daemon_started_ = false;

void CudaMps::startDaemon(int percentage) {
    if (daemon_started_) {
        return;  // 既に起動済み
    }
    
    // CUDA_MPS_PIPE_DIRECTORY環境変数を設定
    std::string pipe_dir = "/tmp/nvidia-mps";
    setenv("CUDA_MPS_PIPE_DIRECTORY", pipe_dir.c_str(), 1);
    
    // MPSコントロールデーモンを起動
    int result = system("nvidia-cuda-mps-control -d");
    if (result != 0) {
        throw std::runtime_error("Failed to start MPS control daemon");
    }
    
    // GPU使用率制限を設定
    if (percentage > 0 && percentage <= 100) {
        std::string cmd = "echo set_default_active_thread_percentage " + 
                         std::to_string(percentage) + " | nvidia-cuda-mps-control";
        system(cmd.c_str());
    }
    
    daemon_started_ = true;
}

void CudaMps::stopDaemon() {
    if (!daemon_started_) {
        return;
    }
    
    // MPSデーモンを停止
    system("echo quit | nvidia-cuda-mps-control");
    daemon_started_ = false;
}

bool CudaMps::isEnabled() {
    // CUDA_MPS_PIPE_DIRECTORY環境変数の存在確認
    const char* pipe_dir = getenv("CUDA_MPS_PIPE_DIRECTORY");
    return (pipe_dir != nullptr) && daemon_started_;
}

std::string CudaMps::getStatus() {
    if (!daemon_started_) {
        return "MPS daemon is not running";
    }
    
    // MPSの状態を取得（簡易実装）
    FILE* fp = popen("echo get_server_list | nvidia-cuda-mps-control 2>/dev/null", "r");
    if (fp == nullptr) {
        return "Unable to get MPS status";
    }
    
    char buffer[256];
    std::string result;
    while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        result += buffer;
    }
    pclose(fp);
    
    if (result.empty()) {
        return "MPS daemon running (no active servers)";
    }
    return "MPS daemon running: " + result;
}

void CudaMps::setActiveThreadPercentage(int percentage) {
    if (!daemon_started_) {
        throw std::runtime_error("MPS daemon is not running");
    }
    
    if (percentage < 0 || percentage > 100) {
        throw std::runtime_error("Invalid percentage value (must be 0-100)");
    }
    
    std::string cmd = "echo set_default_active_thread_percentage " + 
                     std::to_string(percentage) + " | nvidia-cuda-mps-control";
    int result = system(cmd.c_str());
    if (result != 0) {
        throw std::runtime_error("Failed to set active thread percentage");
    }
}

// MpsExecution implementation
MpsExecution::MpsExecution(bool enable_mps, int gpu_percentage) 
    : mps_enabled_(enable_mps), started_daemon_(false) {
    if (mps_enabled_) {
        try {
            if (!CudaMps::isEnabled()) {
                CudaMps::startDaemon(gpu_percentage);
                started_daemon_ = true;
            }
        } catch (const std::exception& e) {
            // MPS起動に失敗した場合は警告を出してMPSなしで続行
            std::cerr << "Warning: Failed to start MPS daemon: " << e.what() << std::endl;
            std::cerr << "Continuing without MPS..." << std::endl;
            mps_enabled_ = false;
        }
    }
}

MpsExecution::~MpsExecution() {
    if (started_daemon_) {
        try {
            CudaMps::stopDaemon();
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to stop MPS daemon: " << e.what() << std::endl;
        }
    }
}

}  // namespace cuda_utils
