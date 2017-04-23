cmake_minimum_required(VERSION 3.0)
project(cudnn_proj)

find_package(CUDA REQUIRED)
find_library(CUDNN_LIB cudnn)

# Set it by yourself, if your cuDNN installation path
# is different from CUDA installation path
set(CUDNN_PATH ${CUDA_TOOLKIT_ROOT_DIR})

include(FindCUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

list(APPEND CUDA_NVCC_FLAGS
        "-arch=sm_30;
        -std=c++11;
        -cudart=shared;")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG")
    list(APPEND CUDA_NVCC_FLAGS
            "-g
            -O0")

else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror")
    list(APPEND CUDA_NVCC_FLAGS
            "-O3;")
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
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}
        )

target_link_libraries(cudnn_proj cudnn)
