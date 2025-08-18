# cuda-scheduling

## requirements
- NVIDIA GPU with CUDA support
- See [`Dockerfile`](.devcontainer/Dockerfile) for additional dependencies

## Run in Docker

```bash
docker build -t cuda-scheduling -f .devcontainer/Dockerfile .
docker run --gpus all -it --rm -v $(pwd):/workspace cuda-scheduling
```

## build project

```bash
meson setup builddir
cd builddir
meson compile
```

## run tests

```bash
cd builddir
meson test
```

Available test suites:
- `cuda_utils_test`: Tests for CUDA utilities (memory management, streams)
- `matrixmul_test`: Tests for matrix multiplication kernels including:
  - Basic kernel correctness tests
  - Basic kernel variant
  - Edge cases (identity matrix, zero matrix)
  - Performance comparison
- `mps_test`: Tests for CUDA Multi-Process Service (MPS) functionality

To run specific test suites:
```bash
meson test cuda_utils_test -v
meson test matrixmul_test -v
meson test mps_test -v
```

## run example

```bash
# Basic execution
./build/src/main

# With MPS support for concurrent execution
./build/src/main --use_mps=true --use_streams=true --num_async=4

# Run comprehensive MPS tests
./scripts/mps_test.sh
```

## CUDA MPS (Multi-Process Service) Support

This project includes support for CUDA Multi-Process Service (MPS), which enables multiple CUDA applications to efficiently share a single GPU.

### Key Features

- **Concurrent Execution**: Multiple processes can run simultaneously on the same GPU
- **Resource Sharing**: Efficient GPU resource utilization across processes
- **Performance Optimization**: Reduced scheduling overhead and improved memory bandwidth usage

### Quick Start with MPS

```bash
# Start MPS daemon
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

# Run with MPS
./main --use_mps=true

# Stop MPS daemon
echo quit | nvidia-cuda-mps-control
```

For detailed MPS usage instructions, see [docs/MPS_USAGE.md](docs/MPS_USAGE.md).

