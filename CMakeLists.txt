cmake_minimum_required(VERSION 3.0)
project(cudnn_proj)

find_package(CUDA REQUIRED)
find_library(CUDNN_LIB cudnn)

# Set it by yourself, if your cuDNN installation path
# is different from CUDA installation path
set(CUDNN_PATH ${CUDA_TOOLKIT_ROOT_DIR})

include(FindCUDA)

list(APPEND CUDA_NVCC_FLAGS
        "-arch=sm_50;
        -std=c++11;
        -O2;
        -cudart=shared;")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CUDNN_PATH} ${CUDNN_PATH}/include)
link_directories(${CUDNN_PATH} ${CUDNN_PATH}/lib ${CUDNN_PATH}/lib64)

set(SOURCE_DIR source)
set(INCLUDE_DIR include)

file(GLOB SOURCES
        "${SOURCE_DIR}/*.cu"
        "${SOURCE_DIR}/*.cpp"
        "${SOURCE_DIR}/*.c")

file(GLOB INCLUDE
        "${INCLUDE_DIR}/*.h"
        "${INCLUDE_DIR}/*.hpp")

include_directories(${INCLUDE_DIR})

cuda_add_executable(cudnn_proj ${SOURCES} ${INCLUDE})
cuda_add_cublas_to_target(cudnn_proj)

set_target_properties(cudnn_proj PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_SOURCE_DIR}/build
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/build
        )

target_link_libraries(cudnn_proj cudnn)