#!/bin/bash

# MPS機能を使った行列乗算のベンチマークスクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/builddir"
RESULTS_DIR="$PROJECT_DIR/results"

# ビルドディレクトリの確認
if [ ! -f "$BUILD_DIR/src/main" ]; then
    echo "Error: main executable not found. Please build the project first."
    echo "Run: cd $PROJECT_DIR && meson compile -C builddir"
    exit 1
fi

# 結果ディレクトリの作成
mkdir -p "$RESULTS_DIR"

echo "=== CUDA MPS Benchmark ==="
echo "Starting MPS benchmarks..."

# テスト設定
MATRIX_SIZES=(512 1024 2048)
NUM_ASYNC_VALUES=(2 4 8)
KERNEL_TYPES=("basic" "tiling" "shared")
MPS_PERCENTAGES=(0 25 50 75 100)

# 結果ファイル
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MPS_RESULTS="$RESULTS_DIR/mps_benchmark_${TIMESTAMP}.csv"

# CSVヘッダーの作成
echo "matrix_size,num_async,kernel_type,use_mps,mps_percentage,cpu_time_ms,cuda_time_ms,avg_kernel_time_ms,gflops_cpu,gflops_cuda" > "$MPS_RESULTS"

echo "Results will be saved to: $MPS_RESULTS"
echo ""

total_tests=0
current_test=0

# 総テスト数の計算
for matrix_size in "${MATRIX_SIZES[@]}"; do
    for num_async in "${NUM_ASYNC_VALUES[@]}"; do
        for kernel_type in "${KERNEL_TYPES[@]}"; do
            # MPS無効
            ((total_tests++))
            # MPS有効（各パーセンテージ）
            for mps_percentage in "${MPS_PERCENTAGES[@]}"; do
                ((total_tests++))
            done
        done
    done
done

echo "Total tests to run: $total_tests"
echo ""

# ベンチマーク実行関数
run_benchmark() {
    local matrix_size=$1
    local num_async=$2
    local kernel_type=$3
    local use_mps=$4
    local mps_percentage=$5
    
    ((current_test++))
    echo "[$current_test/$total_tests] Matrix=$matrix_size, Async=$num_async, Kernel=$kernel_type, MPS=$use_mps"
    if [ "$use_mps" = "true" ]; then
        echo "                MPS percentage: $mps_percentage%"
    fi
    
    # 一時レポートファイル
    local temp_report="/tmp/mps_benchmark_temp_${current_test}.txt"
    
    # プログラム実行
    local cmd="$BUILD_DIR/src/main --matrix_size=$matrix_size --num_async=$num_async --kernel_type=$kernel_type --use_streams=true"
    if [ "$use_mps" = "true" ]; then
        cmd="$cmd --use_mps=true --mps_percentage=$mps_percentage"
    else
        cmd="$cmd --use_mps=false"
    fi
    cmd="$cmd --performance_report=$temp_report"
    
    # 実行とエラーハンドリング
    if timeout 60 $cmd > /dev/null 2>&1; then
        # レポートファイルから結果を抽出
        if [ -f "$temp_report" ]; then
            local cpu_time=$(grep "Total CPU execution time:" "$temp_report" | grep -o '[0-9.]*' | head -1)
            local cuda_time=$(grep "Total CUDA execution time:" "$temp_report" | grep -o '[0-9.]*' | head -1)
            local avg_kernel_time=$(grep "Average kernel time:" "$temp_report" | grep -o '[0-9.]*' | head -1)
            local gflops_cpu=$(grep "Throughput (CPU time):" "$temp_report" | grep -o '[0-9.]*' | head -1)
            local gflops_cuda=$(grep "Throughput (CUDA time):" "$temp_report" | grep -o '[0-9.]*' | head -1)
            
            # CSVに追加
            echo "$matrix_size,$num_async,$kernel_type,$use_mps,$mps_percentage,$cpu_time,$cuda_time,$avg_kernel_time,$gflops_cpu,$gflops_cuda" >> "$MPS_RESULTS"
            
            echo "  CPU time: ${cpu_time}ms, CUDA time: ${cuda_time}ms, GFLOPS: ${gflops_cuda}"
        else
            echo "  Error: Report file not generated"
            echo "$matrix_size,$num_async,$kernel_type,$use_mps,$mps_percentage,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$MPS_RESULTS"
        fi
        
        # 一時ファイル削除
        rm -f "$temp_report"
    else
        echo "  Error: Test timed out or failed"
        echo "$matrix_size,$num_async,$kernel_type,$use_mps,$mps_percentage,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "$MPS_RESULTS"
    fi
    
    # 少し待機（GPUのクールダウン）
    sleep 1
}

# ベンチマーク実行
for matrix_size in "${MATRIX_SIZES[@]}"; do
    for num_async in "${NUM_ASYNC_VALUES[@]}"; do
        for kernel_type in "${KERNEL_TYPES[@]}"; do
            echo ""
            echo "--- Testing: Matrix=$matrix_size, Async=$num_async, Kernel=$kernel_type ---"
            
            # MPS無効テスト
            run_benchmark "$matrix_size" "$num_async" "$kernel_type" "false" "0"
            
            # MPS有効テスト（各パーセンテージ）
            for mps_percentage in "${MPS_PERCENTAGES[@]}"; do
                run_benchmark "$matrix_size" "$num_async" "$kernel_type" "true" "$mps_percentage"
            done
        done
    done
done

echo ""
echo "=== Benchmark Completed ==="
echo "Results saved to: $MPS_RESULTS"
echo ""

# 簡単な結果サマリー
echo "=== Results Summary ==="
echo "Tests with errors:"
grep -c "ERROR\|TIMEOUT" "$MPS_RESULTS" || echo "0"

echo ""
echo "Best GFLOPS (CUDA time) results:"
echo "Without MPS:"
grep ",false," "$MPS_RESULTS" | sort -t',' -k10 -nr | head -3 | while IFS=',' read -r matrix_size num_async kernel_type use_mps mps_percentage cpu_time cuda_time avg_kernel_time gflops_cpu gflops_cuda; do
    echo "  Matrix: $matrix_size, Async: $num_async, Kernel: $kernel_type, GFLOPS: $gflops_cuda"
done

echo ""
echo "With MPS:"
grep ",true," "$MPS_RESULTS" | sort -t',' -k10 -nr | head -3 | while IFS=',' read -r matrix_size num_async kernel_type use_mps mps_percentage cpu_time cuda_time avg_kernel_time gflops_cpu gflops_cuda; do
    echo "  Matrix: $matrix_size, Async: $num_async, Kernel: $kernel_type, MPS: ${mps_percentage}%, GFLOPS: $gflops_cuda"
done

echo ""
echo "For detailed analysis, open: $MPS_RESULTS"
