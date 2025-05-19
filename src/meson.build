matrixmul_files = [
    'matrixmul.cu',
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
    'main.cu',
    dependencies: [cuda],
    link_with: matrixmul,
    install: true,
)
