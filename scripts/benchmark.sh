#!/bin/bash
# パフォーマンス測定スクリプト - CSVレポート生成
set -e

# パラメータ設定
declare -A matrix_sizes=([tiny0]=16 [tiny1]=32 [tiny2]=64 \
  [small0]=128 [small1]=192 [small2]=256 \
  [medium0]=384 [medium1]=512 [medium2]=768 \
  [large0]=1024 [large1]=1536 [large2]=2048 \
  [huge0]=3072 [huge1]=4096 [huge2]=8192
)
num_asyncs=(1 2 4 8)
use_streams_opts=(true false)
kernel_type="basic"

BIN="$(dirname "$0")/../builddir/src/main"

if [ ! -f "$BIN" ]; then
  echo "Error: Binary $BIN not found. Please build the project first."
  exit 1
fi

RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

CSV_FILE="$RESULTS_DIR/performance_benchmark.csv"

# CSVヘッダーを作成
echo "size_name,matrix_size,num_async,use_streams,total_cpu_time,total_cuda_time,sum_kernel_time,average_kernel_time,kernel_0,kernel_1,kernel_2,kernel_3,kernel_4,kernel_5,kernel_6,kernel_7" > "$CSV_FILE"

echo "Starting performance benchmark..."
echo "Results will be saved to: $CSV_FILE"

for size_name in "${!matrix_sizes[@]}"; do
  N="${matrix_sizes[$size_name]}"
  for num_async in "${num_asyncs[@]}"; do
    for use_streams in "${use_streams_opts[@]}"; do
      echo "Running: size=$size_name($N), num_async=$num_async, streams=$use_streams"
      
      # 一時的なレポートファイル
      TEMP_REPORT="/tmp/temp_report_$$.txt"
      TEMP_OUTPUT="/tmp/temp_output_$$.txt"
      
      # プログラムを実行してレポートと標準出力を生成
      ARGS="--matrix_size=$N --num_async=$num_async --kernel_type=$kernel_type --use_streams=$use_streams --performance_report=$TEMP_REPORT"
      "$BIN" $ARGS > "$TEMP_OUTPUT" 2>&1
      
      # レポートファイルからデータを抽出
      if [ -f "$TEMP_REPORT" ]; then
        total_cpu_time=$(grep "Total CPU execution time:" "$TEMP_REPORT" | awk '{print $5}')
        total_cuda_time=$(grep "Total CUDA execution time:" "$TEMP_REPORT" | awk '{print $5}')
        average_kernel_time=$(grep "Average kernel time:" "$TEMP_REPORT" | awk '{print $4}')
        
        # 個別カーネル時間を抽出
        kernel_times=""
        for i in $(seq 0 $((num_async-1))); do
          kernel_time=$(grep "Kernel $i:" "$TEMP_REPORT" | awk '{print $3}')
          kernel_times="$kernel_times,$kernel_time"
        done
        
        # sum_kernel_timeを標準出力から抽出
        sum_kernel_time=$(grep "Sum of individual kernel times:" "$TEMP_OUTPUT" | awk '{print $6}')
        
        # 不足分は空文字で埋める（最大8カーネルまで対応）
        for i in $(seq $num_async 7); do
          kernel_times="$kernel_times,"
        done
        
        # CSVに行を追加
        echo "$size_name,$N,$num_async,$use_streams,$total_cpu_time,$total_cuda_time,$sum_kernel_time,$average_kernel_time$kernel_times" >> "$CSV_FILE"
        
        rm -f "$TEMP_REPORT" "$TEMP_OUTPUT"
      else
        echo "Warning: Report file not generated for this configuration"
      fi
    done
  done
done

echo "Benchmark completed!"
echo "Results saved to: $CSV_FILE"
echo "CSV columns:"
echo "  size_name: Size configuration name"
echo "  matrix_size: Actual matrix size (N x N)"
echo "  num_async: Number of asynchronous operations"
echo "  use_streams: Whether CUDA streams were used"
echo "  total_cpu_time: Total CPU execution time (ms)"
echo "  total_cuda_time: Total CUDA execution time (ms)"
echo "  sum_kernel_time: Sum of individual kernel times (ms)"
echo "  average_kernel_time: Average kernel execution time (ms)"
echo "  kernel_0-7: Individual kernel execution times (ms)"
