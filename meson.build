project(
    'matrixmul',
    'cpp', 'cuda',
    default_options: [
        'cpp_std=c++17',
        'cuda_std=c++17',
    ]
)

cuda = dependency('cuda', required: true)

# GFlags dependency using subproject wrap
gflags_dep = dependency('gflags', required: true, fallback: ['gflags', 'gflags_dep'])

# Google Test dependency using subproject wrap
gtest_dep = dependency('gtest', required: false, fallback: ['gtest', 'gtest_dep'])

src_files = []

subdir('src')