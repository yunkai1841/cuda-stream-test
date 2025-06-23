matrixmul_files = [
    'matrixmul.cu',
    'launcher.cu',
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
