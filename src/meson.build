matrixmul_files = [
    'matrixmul.cu',
    'launcher.cu',
    'cuda_utils.cpp',
]

src_files += matrixmul_files

matrixmul = shared_library(
    'matrixmul',
    src_files,
    dependencies: [cuda],
    include_directories: include_directories('.'),
    install: true,
)

main_exe = executable(
    'main',
    'main.cpp',
    dependencies: [cuda, gflags_dep],
    link_with: matrixmul,
    install: true,
)

# Test executable for CUDA utils
if gtest_dep.found()
    test_exe = executable(
        'test_cuda_utils',
        'test_cuda_utils.cpp',
        dependencies: [cuda, gtest_dep],
        link_with: matrixmul,
        install: false,
    )
    
    test('cuda_utils_test', test_exe)
endif
