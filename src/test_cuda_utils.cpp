#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "cuda_utils.h"

using namespace cuda_utils;

class CudaUtilsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // CUDA初期化をチェック
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "CUDA device not available";
        }
    }
};

// CudaMemoryテスト
TEST_F(CudaUtilsTest, CudaMemoryBasicAllocation) {
    const size_t size = 1024 * sizeof(float);

    // メモリ確保テスト
    auto memory = std::make_unique<CudaMemory>(size);

    EXPECT_TRUE(*memory);
    EXPECT_NE(memory->get(), nullptr);
    EXPECT_EQ(memory->size(), size);
}

TEST_F(CudaUtilsTest, CudaMemoryZeroAllocation) {
    // ゼロサイズの確保（例外は投げない）
    EXPECT_NO_THROW({
        CudaMemory mem(0);
        EXPECT_FALSE(mem);
        EXPECT_EQ(mem.get(), nullptr);
        EXPECT_EQ(mem.size(), 0);
    });
}

TEST_F(CudaUtilsTest, CudaMemoryMoveConstructor) {
    const size_t size = 512 * sizeof(float);

    CudaMemory memory1(size);
    float* original_ptr = memory1.get();

    // ムーブコンストラクタ
    CudaMemory memory2(std::move(memory1));

    // memory1は空になり、memory2が所有権を持つ
    EXPECT_FALSE(memory1);
    EXPECT_EQ(memory1.get(), nullptr);
    EXPECT_EQ(memory1.size(), 0);

    EXPECT_TRUE(memory2);
    EXPECT_EQ(memory2.get(), original_ptr);
    EXPECT_EQ(memory2.size(), size);
}

TEST_F(CudaUtilsTest, CudaMemoryMoveAssignment) {
    const size_t size1 = 256 * sizeof(float);
    const size_t size2 = 512 * sizeof(float);

    CudaMemory memory1(size1);
    CudaMemory memory2(size2);

    float* ptr2 = memory2.get();

    // ムーブ代入
    memory1 = std::move(memory2);

    EXPECT_EQ(memory1.get(), ptr2);
    EXPECT_EQ(memory1.size(), size2);

    EXPECT_FALSE(memory2);
    EXPECT_EQ(memory2.get(), nullptr);
    EXPECT_EQ(memory2.size(), 0);
}

TEST_F(CudaUtilsTest, CudaMemoryRelease) {
    const size_t size = 128 * sizeof(float);

    CudaMemory memory(size);
    float* ptr = memory.get();

    // 所有権の放棄
    float* released_ptr = memory.release();

    EXPECT_EQ(released_ptr, ptr);
    EXPECT_FALSE(memory);
    EXPECT_EQ(memory.get(), nullptr);
    EXPECT_EQ(memory.size(), 0);

    // 手動で解放
    cudaFree(released_ptr);
}

TEST_F(CudaUtilsTest, CudaMemoryReset) {
    const size_t size = 256 * sizeof(float);

    CudaMemory memory(size);

    // nullptrでリセット
    memory.reset();

    EXPECT_FALSE(memory);
    EXPECT_EQ(memory.get(), nullptr);
    EXPECT_EQ(memory.size(), 0);
}

TEST_F(CudaUtilsTest, CudaMemoryDataTransfer) {
    const size_t size = 100;
    const size_t bytes = size * sizeof(float);

    // ホストメモリ
    std::vector<float> host_data(size, 3.14f);
    std::vector<float> result_data(size, 0.0f);

    // デバイスメモリ
    CudaMemory device_memory(bytes);

    // ホスト→デバイス転送
    cudaError_t err =
        cudaMemcpy(device_memory.get(), host_data.data(), bytes, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);

    // デバイス→ホスト転送
    err = cudaMemcpy(result_data.data(), device_memory.get(), bytes, cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);

    // データの検証
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result_data[i], 3.14f);
    }
}

// CudaStreamテスト
TEST_F(CudaUtilsTest, CudaStreamBasicCreation) {
    auto stream = std::make_unique<CudaStream>();

    EXPECT_NE(stream->get(), nullptr);
}

TEST_F(CudaUtilsTest, CudaStreamMoveConstructor) {
    CudaStream stream1;
    cudaStream_t original_stream = stream1.get();

    // ムーブコンストラクタ
    CudaStream stream2(std::move(stream1));

    EXPECT_EQ(stream2.get(), original_stream);
    EXPECT_EQ(stream1.get(), nullptr);
}

TEST_F(CudaUtilsTest, CudaStreamMoveAssignment) {
    CudaStream stream1;
    CudaStream stream2;

    cudaStream_t stream2_handle = stream2.get();

    // ムーブ代入
    stream1 = std::move(stream2);

    EXPECT_EQ(stream1.get(), stream2_handle);
    EXPECT_EQ(stream2.get(), nullptr);
}

TEST_F(CudaUtilsTest, CudaStreamSynchronization) {
    CudaStream stream;

    // 同期テスト（例外が発生しないかチェック）
    EXPECT_NO_THROW(stream.synchronize());
}

TEST_F(CudaUtilsTest, MultipleStreamsIndependence) {
    const int num_streams = 4;
    std::vector<std::unique_ptr<CudaStream>> streams;

    // 複数のストリームを作成
    for (int i = 0; i < num_streams; ++i) {
        streams.push_back(std::make_unique<CudaStream>());
    }

    // すべて異なるハンドルを持つことを確認
    for (int i = 0; i < num_streams; ++i) {
        for (int j = i + 1; j < num_streams; ++j) {
            EXPECT_NE(streams[i]->get(), streams[j]->get());
        }
    }
}

// RAIIの動作テスト
TEST_F(CudaUtilsTest, RAIIAutoCleanup) {
    // スコープ内でメモリとストリームを作成
    {
        auto memory = std::make_unique<CudaMemory>(1024 * sizeof(float));
        auto stream = std::make_unique<CudaStream>();

        EXPECT_TRUE(*memory);
        EXPECT_NE(stream->get(), nullptr);

        // スコープを抜ける際に自動的にクリーンアップされる
    }

    // ここでメモリリークがないことを期待
    // （実際のメモリリークチェックは外部ツールが必要）
    SUCCEED();
}

// エラーハンドリングテスト
TEST_F(CudaUtilsTest, ErrorHandling) {
    // 極端に大きなメモリ確保（失敗するはず）
    const size_t huge_size = SIZE_MAX;

    EXPECT_THROW({ CudaMemory memory(huge_size); }, std::runtime_error);
}
