cmake_minimum_required(VERSION 3.0)
project(cudnn_proj)

find_package(CUDA REQUIRED)
find_library(CUDNN_LIB cudnn)

set(CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/build")

# Set it by yourself, if your cuDNN installation path
# is different from CUDA installation path
set(CUDNN_PATH ${CUDA_TOOLKIT_ROOT_DIR})

include(FindCUDA)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3 -Wall -Werror")

list(APPEND CUDA_NVCC_FLAGS
        "-arch=sm_50;
        -std=c++11;
        -cudart=shared;")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS
            "-g;
            -O0;")
else()
    list(APPEND CUDA_NVCC_FLAGS
            "
            -O3;")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")


set(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(${CUDNN_PATH} ${CUDNN_PATH}/include)
link_directories(${CUDNN_PATH} ${CUDNN_PATH}/lib ${CUDNN_PATH}/lib64)

set(SOURCE_DIR source)
set(INCLUDE_DIR include)

file(GLOB SOURCES
        "${SOURCE_DIR}/*.cu"
        "${SOURCE_DIR}/*.cpp"
        "${SOURCE_DIR}/*.c")

file(GLOB INCLUDE
        "${INCLUDE_DIR}/*.cuh"
        "${INCLUDE_DIR}/*.h"
        "${INCLUDE_DIR}/*.hpp")

include_directories(${INCLUDE_DIR})

cuda_add_executable(cudnn_proj ${SOURCES} ${INCLUDE})
cuda_add_cublas_to_target(cudnn_proj)

set_target_properties(cudnn_proj PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/${CMAKE_BUILD_TYPE}
        )

target_link_libraries(cudnn_proj cudnn)