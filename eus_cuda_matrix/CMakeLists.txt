cmake_minimum_required(VERSION 2.8.3)
project(eus_cuda_matrix)

find_package(CUDA REQUIRED)
#set(CUDA_SEPARABLE_COMPILATION ON) # separate compile is very difficult ...

include_directories("/usr/local/cuda/samples/common/inc")

set(CUDA_NVCC_FLAGS "-arch=sm_50; -Xcompiler -fPIC")
cuda_compile(matrixMul_obj src/matrixMul.cu)
cuda_compile(matrixMulCUBLAS_obj src/matrixMulCUBLAS.cu)

cuda_add_library(eus_cuda_matrix SHARED ${matrixMul_obj} ${matrixMulCUBLAS_obj})
cuda_add_cublas_to_target(eus_cuda_matrix)

add_executable(test_matrixMul src/test/test_matrixMul.c)
target_link_libraries(test_matrixMul eus_cuda_matrix)

add_executable(test_matrixMulCUBLAS src/test/test_matrixMulCUBLAS.c)
target_link_libraries(test_matrixMulCUBLAS eus_cuda_matrix)
