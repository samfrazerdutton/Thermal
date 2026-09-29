set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(thermal_cuda_kernels STATIC
    vector_add.cu
    reduction.cu
    matmul.cu
)
target_include_directories(thermal_cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(thermal_cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(thermal_kernel_bench bench_main.cu)
target_link_libraries(thermal_kernel_bench PRIVATE thermal_cuda_kernels)
