project(
    'matrixmul',
    'cpp', 'cuda',
)

cuda = dependency('cuda', required: true)

cmake = import('cmake')
gflags_proj = cmake.subproject('gflags')
message('CMake targets:\n - ' + '\n - '.join(gflags_proj.target_list()))
gflags_dep = gflags_proj.dependency('gflags_static')

src_files = []

subdir('src')