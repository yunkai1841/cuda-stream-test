#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cuda_utils {

/**
 * @brief CUDAメモリ管理のためのRAIIラッパークラス
 * 
 * std::unique_ptrのようなインターフェースを提供し、
 * CUDAメモリの自動管理を行います。
 */
class CudaMemory {
public:
    /**
     * @brief 指定されたサイズのCUDAメモリを確保
     * @param size 確保するメモリサイズ（バイト）
     * @throws std::runtime_error メモリ確保に失敗した場合
     */
    explicit CudaMemory(size_t size);
    
    /**
     * @brief デストラクタ - CUDAメモリを自動解放
     */
    ~CudaMemory();
    
    // コピーコンストラクタとコピー代入演算子を削除（unique_ptr風）
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    
    // ムーブコンストラクタとムーブ代入演算子
    CudaMemory(CudaMemory&& other) noexcept;
    CudaMemory& operator=(CudaMemory&& other) noexcept;
    
    /**
     * @brief 管理しているメモリポインタを取得
     * @return CUDAメモリポインタ
     */
    float* get() const { return ptr_; }
    
    /**
     * @brief 確保されているメモリサイズを取得
     * @return メモリサイズ（バイト）
     */
    size_t size() const { return size_; }
    
    /**
     * @brief メモリが確保されているかチェック
     * @return メモリが確保されている場合true
     */
    explicit operator bool() const { return ptr_ != nullptr; }
    
    /**
     * @brief 生ポインタの所有権を放棄
     * @return 管理していたポインタ（呼び出し側で解放が必要）
     */
    float* release();
    
    /**
     * @brief 新しいメモリを管理対象に設定
     * @param new_ptr 新しく管理するポインタ
     * @param new_size 新しいメモリサイズ
     */
    void reset(float* new_ptr = nullptr, size_t new_size = 0);
    
private:
    float* ptr_;
    size_t size_;
};

/**
 * @brief CUDAストリーム管理のためのRAIIラッパークラス
 * 
 * CUDAストリームの自動作成・破棄を行います。
 */
class CudaStream {
public:
    /**
     * @brief CUDAストリームを作成
     * @throws std::runtime_error ストリーム作成に失敗した場合
     */
    CudaStream();
    
    /**
     * @brief デストラクタ - CUDAストリームを自動破棄
     */
    ~CudaStream();
    
    // コピー禁止
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // ムーブ可能
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;
    
    /**
     * @brief CUDAストリームハンドルを取得
     * @return cudaStream_tハンドル
     */
    cudaStream_t get() const { return stream_; }
    
    /**
     * @brief ストリームの同期を行う
     */
    void synchronize() const;
    
private:
    cudaStream_t stream_;
};

} // namespace cuda_utils
