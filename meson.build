project(
    'matrixmul',
    'cpp', 'cuda',
    default_options: [
        'cpp_std=c++17',
        'cuda_std=c++17',
    ]
)

cuda = dependency('cuda', required: true)

# GFlags dependency using specific version 2.2.2
gflags_dep = dependency('gflags', version: '==2.2.2', required: true, fallback: ['gflags', 'gflags_dep'])

# Google Test dependency using specific version 1.17.0
gtest_dep = dependency('gtest', required: false, fallback: ['gtest', 'gtest_dep'])
gtest_main_dep = dependency('gtest_main', required: false, fallback: ['gtest', 'gtest_main_dep'])

src_files = []

subdir('src')