project(
    'matrixmul',
    'cuda',
)

cuda = dependency('cuda', required: true)

executable(
    'matrixmul',
    'matrixmul.cu',
    dependencies: cuda,
    install: true
)