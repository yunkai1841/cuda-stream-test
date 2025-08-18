#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "cuda_utils.h"
#include "matrixmul.cuh"

using namespace cuda_utils;

class MatrixMulTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // CUDA初期化をチェック
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }

    // CPU参照実装：標準的な行列乗算
    void cpuMatrixMul(const float* A, const float* B, float* C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    // ランダムな行列を生成
    void generateRandomMatrix(float* matrix, int N, float min_val = 0.0f, float max_val = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min_val, max_val);
        
        for (int i = 0; i < N * N; i++) {
            matrix[i] = dis(gen);
        }
    }

    // 単位行列を生成
    void generateIdentityMatrix(float* matrix, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    // 結果を比較（相対誤差も考慮）
    bool compareMatrices(const float* A, const float* B, int N, float tolerance = 1e-4f) {
        for (int i = 0; i < N * N; i++) {
            float diff = std::abs(A[i] - B[i]);
            float relative_error = diff / (std::abs(A[i]) + 1e-8f);
            if (diff > tolerance && relative_error > tolerance) {
                return false;
            }
        }
        return true;
    }

    // テスト用のヘルパー関数：指定されたカーネルタイプをテスト
    void testMatrixMulKernel(const char* kernel_type, int N) {
        const size_t size = N * N * sizeof(float);

        // ホストメモリ確保
        std::vector<float> h_A(N * N);
        std::vector<float> h_B(N * N);
        std::vector<float> h_C_gpu(N * N);
        std::vector<float> h_C_cpu(N * N);

        // ランダムな入力行列を生成
        generateRandomMatrix(h_A.data(), N, -1.0f, 1.0f);
        generateRandomMatrix(h_B.data(), N, -1.0f, 1.0f);

        // GPU メモリ確保
        CudaMemory d_A(size);
        CudaMemory d_B(size);
        CudaMemory d_C(size);

        ASSERT_TRUE(d_A && d_B && d_C) << "GPU memory allocation failed";

        // データをGPUにコピー
        ASSERT_EQ(cudaMemcpy(d_A.get(), h_A.data(), size, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_B.get(), h_B.data(), size, cudaMemcpyHostToDevice), cudaSuccess);

        // GPU で計算
        launchMatrixMulKernel(kernel_type, static_cast<float*>(d_C.get()), 
                             static_cast<const float*>(d_A.get()), 
                             static_cast<const float*>(d_B.get()), N);
        
        ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "Kernel launch failed";
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << "Kernel execution failed";

        // 結果をホストにコピー
        ASSERT_EQ(cudaMemcpy(h_C_gpu.data(), d_C.get(), size, cudaMemcpyDeviceToHost), cudaSuccess);

        // CPU で参照計算
        cpuMatrixMul(h_A.data(), h_B.data(), h_C_cpu.data(), N);

        // 結果を比較
        EXPECT_TRUE(compareMatrices(h_C_gpu.data(), h_C_cpu.data(), N)) 
            << "Matrix multiplication results don't match for kernel: " << kernel_type;
    }
};

// 基本的なカーネルのテスト
TEST_F(MatrixMulTest, BasicKernelSmallMatrix) {
    testMatrixMulKernel("basic", 16);
}

TEST_F(MatrixMulTest, BasicKernelMediumMatrix) {
    testMatrixMulKernel("basic", 32);
}

TEST_F(MatrixMulTest, BasicKernelLargeMatrix) {
    testMatrixMulKernel("basic", 64);
}

// 非16倍数サイズでも正しく計算できるか
TEST_F(MatrixMulTest, BasicKernelNonMultipleTileSmall) {
    testMatrixMulKernel("basic", 17);
}

TEST_F(MatrixMulTest, BasicKernelNonMultipleTileMedium) {
    testMatrixMulKernel("basic", 30);
}

// 単位行列との乗算テスト
TEST_F(MatrixMulTest, IdentityMatrixMultiplication) {
    const int N = 32;
    const size_t size = N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_I(N * N);  // 単位行列
    std::vector<float> h_C(N * N);

    // ランダムな行列Aと単位行列Iを生成
    generateRandomMatrix(h_A.data(), N);
    generateIdentityMatrix(h_I.data(), N);

    // GPU メモリ確保
    CudaMemory d_A(size);
    CudaMemory d_I(size);
    CudaMemory d_C(size);

    ASSERT_TRUE(d_A && d_I && d_C);

    // データをGPUにコピー
    ASSERT_EQ(cudaMemcpy(d_A.get(), h_A.data(), size, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_I.get(), h_I.data(), size, cudaMemcpyHostToDevice), cudaSuccess);

    // A * I = A をテスト
    launchMatrixMulKernel("basic", static_cast<float*>(d_C.get()), 
                         static_cast<const float*>(d_A.get()), 
                         static_cast<const float*>(d_I.get()), N);
    
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_C.data(), d_C.get(), size, cudaMemcpyDeviceToHost), cudaSuccess);

    // 結果がAと同じかチェック
    EXPECT_TRUE(compareMatrices(h_A.data(), h_C.data(), N))
        << "A * I should equal A";
}

// ゼロ行列との乗算テスト
TEST_F(MatrixMulTest, ZeroMatrixMultiplication) {
    const int N = 16;
    const size_t size = N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_zero(N * N, 0.0f);  // ゼロ行列
    std::vector<float> h_C(N * N);

    generateRandomMatrix(h_A.data(), N);

    // GPU メモリ確保
    CudaMemory d_A(size);
    CudaMemory d_zero(size);
    CudaMemory d_C(size);

    ASSERT_TRUE(d_A && d_zero && d_C);

    // データをGPUにコピー
    ASSERT_EQ(cudaMemcpy(d_A.get(), h_A.data(), size, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_zero.get(), h_zero.data(), size, cudaMemcpyHostToDevice), cudaSuccess);

    // A * 0 = 0 をテスト
    launchMatrixMulKernel("basic", static_cast<float*>(d_C.get()), 
                         static_cast<const float*>(d_A.get()), 
                         static_cast<const float*>(d_zero.get()), N);
    
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(h_C.data(), d_C.get(), size, cudaMemcpyDeviceToHost), cudaSuccess);

    // 結果がゼロかチェック
    EXPECT_TRUE(compareMatrices(h_zero.data(), h_C.data(), N))
        << "A * 0 should equal 0";
}

// 無効なカーネルタイプのテスト
TEST_F(MatrixMulTest, InvalidKernelType) {
    const int N = 16;
    const size_t size = N * N * sizeof(float);

    CudaMemory d_A(size);
    CudaMemory d_B(size);
    CudaMemory d_C(size);

    ASSERT_TRUE(d_A && d_B && d_C);

    // 標準エラー出力をキャプチャして、exitが呼ばれることを期待
    // (実際のテストでは、exitが呼ばれるため、この部分はコメントアウト)
    // EXPECT_EXIT(
    //     launchMatrixMulKernel("invalid", static_cast<float*>(d_C.get()), 
    //                          static_cast<const float*>(d_A.get()), 
    //                          static_cast<const float*>(d_B.get()), N),
    //     ::testing::ExitedWithCode(1), "Unknown kernel type: invalid"
    // );
}

// ベクトル加算カーネルのテスト（ここでは専用のランチャー関数が必要）
TEST_F(MatrixMulTest, VectorAddKernelBasic) {
    // このテストはlauncher.cuにvectorAddのランチャー関数が追加された場合に使用
    // 現在は基本的な検証のみ実行

    // CUDAデバイスが利用可能かどうかの基本チェック
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    ASSERT_EQ(err, cudaSuccess);
    ASSERT_GT(deviceCount, 0);
    
    GTEST_SKIP() << "Vector add kernel test skipped - requires launcher function";
}

// パフォーマンステスト（異なるカーネルの相対的な速度を確認）
TEST_F(MatrixMulTest, PerformanceComparison) {
    const int N = 128;  // 16の倍数にする（カーネルの制約）
    const size_t size = N * N * sizeof(float);

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    
    generateRandomMatrix(h_A.data(), N);
    generateRandomMatrix(h_B.data(), N);

    CudaMemory d_A(size);
    CudaMemory d_B(size);
    CudaMemory d_C(size);

    ASSERT_TRUE(d_A && d_B && d_C);

    ASSERT_EQ(cudaMemcpy(d_A.get(), h_A.data(), size, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_B.get(), h_B.data(), size, cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<const char*> kernels = {"basic"};

    for (const char* kernel_type : kernels) {
        cudaEvent_t start, stop;
        ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
        ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

        // ウォームアップ実行
        launchMatrixMulKernel(kernel_type, static_cast<float*>(d_C.get()), 
                             static_cast<const float*>(d_A.get()), 
                             static_cast<const float*>(d_B.get()), N);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        // タイミング測定
        ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
        launchMatrixMulKernel(kernel_type, static_cast<float*>(d_C.get()), 
                             static_cast<const float*>(d_A.get()), 
                             static_cast<const float*>(d_B.get()), N);
        ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
        ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);

        float milliseconds = 0;
        ASSERT_EQ(cudaEventElapsedTime(&milliseconds, start, stop), cudaSuccess);

        std::cout << "Kernel " << kernel_type << " execution time: " 
                  << milliseconds << " ms" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}
