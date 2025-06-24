#!/bin/bash
# Nsight Systems で main バイナリを全パラメータ組み合わせでプロファイルするスクリプト
set -e

# matrix_size（連想配列として宣言）
declare -A sizes=([small]=256 [medium]=512 [large]=1024 [huge]=2048)
# num_asyncs
num_asyncs=(1 2 4 8)
# kernel_type
kernel_types=(basic tiling shared unroll)
# use_streams
use_streams_opts=(true false)

BIN="$(dirname "$0")/../builddir/src/main"

if [ ! -f "$BIN" ]; then
  echo "Error: Binary $BIN not found. Please build the project first."
  exit 1
fi

RESULTS_DIR="$(dirname "$0")/../results"
mkdir -p "$RESULTS_DIR"

for size_name in "${!sizes[@]}"; do
  N="${sizes[$size_name]}"
  for num_async in "${num_asyncs[@]}"; do
    for kernel_type in "${kernel_types[@]}"; do
      for use_streams in "${use_streams_opts[@]}"; do
        OUT_FILE="$RESULTS_DIR/nsys_${size_name}_n${num_async}_${kernel_type}_streams${use_streams}.nsys-rep"
        ARGS="--matrix_size=$N --num_async=$num_async --kernel_type=$kernel_type --use_streams=$use_streams"
        echo "Profiling: $ARGS -> $OUT_FILE"
        nsys profile -o "$OUT_FILE" "$BIN" $ARGS
      done
    done
  done
done

echo "Profiling completed. Output files:"
ls -1 "$RESULTS_DIR"/nsys_*.nsys-rep
