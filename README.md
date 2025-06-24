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
cd build
meson compile
```

## run tests

```bash
meson test
```
## run example

```bash
./builddir/src/main
```

