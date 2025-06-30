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
  - Different kernel variants (basic, tiling, shared memory, unroll)
  - Edge cases (identity matrix, zero matrix)
  - Performance comparison between kernels

To run specific test suites:
```bash
meson test cuda_utils_test -v
meson test matrixmul_test -v
```

## run example

```bash
./build/src/main
```

