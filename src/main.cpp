#include <cuda_runtime.h>
#include <gflags/gflags.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cuda_utils.h"
#include "matrixmul.cuh"

// コマンドライン引数の定義
DEFINE_int32(matrix_size, 1024, "Matrix size (N x N)");
DEFINE_int32(num_async, 4, "Number of asynchronous matrix multiplications");
DEFINE_string(kernel_type, "basic", "Kernel type: basic, tiling, shared, unroll");
DEFINE_bool(use_streams, true, "Use CUDA streams for asynchronous execution");

using namespace cuda_utils;

// 行列の初期化
void initializeMatrix(float* matrix, int N, float value = 1.0f) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = value;
    }
}

// ラッパー関数: std::stringをconst char*に変換してlauncher.cuの関数を呼び出し
void launchMatrixMulKernelWrapper(const std::string& kernel_type, float* d_C, const float* d_A,
                                  const float* d_B, int N, cudaStream_t stream = 0) {
    launchMatrixMulKernel(kernel_type.c_str(), d_C, d_A, d_B, N, stream);
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int N = FLAGS_matrix_size;
    const int num_async = FLAGS_num_async;
    const std::string kernel_type = FLAGS_kernel_type;
    const bool use_streams = FLAGS_use_streams;

    std::cout << "Matrix Multiplication Configuration:" << std::endl;
    std::cout << "  Matrix size: " << N << " x " << N << std::endl;
    std::cout << "  Number of async operations: " << num_async << std::endl;
    std::cout << "  Kernel type: " << kernel_type << std::endl;
    std::cout << "  Use streams: " << (use_streams ? "Yes" : "No") << std::endl;

    // メモリサイズの計算
    const size_t matrix_bytes = N * N * sizeof(float);

    // ホストメモリの確保
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<std::vector<float>> h_C_results(num_async, std::vector<float>(N * N));

    // 行列の初期化
    initializeMatrix(h_A.data(), N, 1.0f);
    initializeMatrix(h_B.data(), N, 2.0f);

    // デバイスメモリの確保（スマートポインタとRAII）
    auto d_A = std::make_unique<CudaMemory>(matrix_bytes);
    auto d_B = std::make_unique<CudaMemory>(matrix_bytes);
    std::vector<std::unique_ptr<CudaMemory>> d_C_array;

    for (int i = 0; i < num_async; i++) {
        d_C_array.push_back(std::make_unique<CudaMemory>(matrix_bytes));
    }

    // CUDAストリームの作成（スマートポインタとRAII）
    std::vector<std::unique_ptr<CudaStream>> streams;
    if (use_streams) {
        for (int i = 0; i < num_async; i++) {
            streams.push_back(std::make_unique<CudaStream>());
        }
    }

    // ホストからデバイスへのデータ転送
    cudaMemcpy(d_A->get(), h_A.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B->get(), h_B.data(), matrix_bytes, cudaMemcpyHostToDevice);

    // 実行時間の測定開始
    auto start = std::chrono::high_resolution_clock::now();

    // 非同期でmatrix multiplicationを実行
    for (int i = 0; i < num_async; i++) {
        cudaStream_t stream = use_streams ? streams[i]->get() : 0;
        launchMatrixMulKernelWrapper(kernel_type, d_C_array[i]->get(), d_A->get(), d_B->get(), N,
                                     stream);
    }

    // すべてのカーネルの完了を待機
    if (use_streams) {
        for (int i = 0; i < num_async; i++) {
            streams[i]->synchronize();
        }
    } else {
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 結果をホストにコピー
    for (int i = 0; i < num_async; i++) {
        cudaStream_t stream = use_streams ? streams[i]->get() : 0;
        cudaMemcpyAsync(h_C_results[i].data(), d_C_array[i]->get(), matrix_bytes,
                        cudaMemcpyDeviceToHost, stream);
    }

    // すべての転送完了を待機
    if (use_streams) {
        for (int i = 0; i < num_async; i++) {
            streams[i]->synchronize();
        }
    } else {
        cudaDeviceSynchronize();
    }

    // 結果の検証（最初の要素のみ）
    std::cout << "\nResults verification:" << std::endl;
    for (int i = 0; i < num_async; i++) {
        std::cout << "  Operation " << i << ": C[0][0] = " << h_C_results[i][0] << std::endl;
    }

    // パフォーマンス結果の表示
    std::cout << "\nPerformance Results:" << std::endl;
    std::cout << "  Total execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Average time per operation: " << duration.count() / num_async << " ms"
              << std::endl;

    // 計算スループットの計算（GFLOPS）
    double flops = 2.0 * N * N * N * num_async;        // 各行列乗算で2*N^3 FLOPs
    double gflops = flops / (duration.count() * 1e6);  // GFLOPS
    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;

    // クリーンアップ（スマートポインタが自動的に行うため、手動は不要）
    // streams と d_C_array は自動的にデストラクタが呼ばれる

    std::cout << "\nMatrix multiplication completed successfully!" << std::endl;
    return 0;
}
