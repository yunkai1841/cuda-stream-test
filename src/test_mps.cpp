#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "cuda_utils.h"

using namespace cuda_utils;

class CudaMPSTest : public ::testing::Test {
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

// MPS利用可能性テスト
TEST_F(CudaMPSTest, MPSAvailabilityCheck) {
    // MPS利用可能性のチェック（デバイス依存）
    bool available = CudaMPS::isAvailable();
    
    // 結果に関わらず、テストは成功（利用可能性はハードウェア依存）
    std::cout << "MPS available: " << (available ? "Yes" : "No") << std::endl;
    SUCCEED();
}

// MPS実行状態テスト
TEST_F(CudaMPSTest, MPSRunningStatusCheck) {
    // MPS実行状態のチェック
    bool running = CudaMPS::isRunning();
    
    // 結果に関わらず、テストは成功（実行状態は環境依存）
    std::cout << "MPS running: " << (running ? "Yes" : "No") << std::endl;
    SUCCEED();
}

// MPSステータス取得テスト
TEST_F(CudaMPSTest, MPSStatusString) {
    std::string status = CudaMPS::getStatus();
    
    // ステータス文字列が空でないことを確認
    EXPECT_FALSE(status.empty());
    EXPECT_TRUE(status.find("MPS Status:") != std::string::npos);
    
    std::cout << "MPS Status:" << std::endl << status << std::endl;
}

// MPS推奨設定表示テスト
TEST_F(CudaMPSTest, MPSRecommendedSettingsDisplay) {
    // 例外が発生しないことを確認
    EXPECT_NO_THROW(CudaMPS::printRecommendedSettings());
}

// MPS機能統合テスト
TEST_F(CudaMPSTest, MPSIntegrationTest) {
    // MPSが利用可能な場合のみテスト実行
    if (!CudaMPS::isAvailable()) {
        GTEST_SKIP() << "MPS not available on this device";
    }
    
    // 利用可能性と実行状態の整合性チェック
    bool available = CudaMPS::isAvailable();
    bool running = CudaMPS::isRunning();
    
    EXPECT_TRUE(available);  // この時点で利用可能なはず
    
    // 実行状態は環境依存のため、エラーを出さずに情報表示
    std::cout << "MPS Integration Test Results:" << std::endl;
    std::cout << "  Available: " << available << std::endl;
    std::cout << "  Running: " << running << std::endl;
    
    if (running) {
        std::cout << "  MPS daemon is active - concurrent execution possible" << std::endl;
    } else {
        std::cout << "  MPS daemon not running - single process execution only" << std::endl;
    }
}
