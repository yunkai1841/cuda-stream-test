project(
    'matrixmul',
    'cpp', 'cuda',
    default_options: [
        'cpp_std=c++17',
        'cuda_std=c++17',
    ]
)

cuda = dependency('cuda', required: true)

cmake = import('cmake')
gflags_proj = cmake.subproject('gflags')
message('CMake targets:\n - ' + '\n - '.join(gflags_proj.target_list()))
gflags_dep = gflags_proj.dependency('gflags_static')

# Google Test dependency using meson wrap
gtest_dep = dependency('gtest', required: false, fallback: ['gtest', 'gtest_main_dep'])

src_files = []

subdir('src')