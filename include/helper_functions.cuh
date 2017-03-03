#ifndef CUDNN_PROJ_HELPER_FUNCTIONS_CUH
#define CUDNN_PROJ_HELPER_FUNCTIONS_CUH

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>

#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>


inline void _checkCudnnErrors(const cudnnStatus_t& status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::stringstream _error;
        _error  << "CUDNN failure: "
                << cudnnGetErrorString(status)
                << std::endl;
        _error << file << ':' << line << std::endl;
        throw std::runtime_error(_error.str().c_str());
    }
}

#define checkCudnnErrors(status) _checkCudnnErrors(status, __FILE__, __LINE__);



inline void _checkCudaErrors(const cudaError_t& status, const char* file, int line) {
    if (status != cudaSuccess) {
        std::stringstream _error;
        _error  << "Cuda failure: "
                << cudaGetErrorString(status)
                << std::endl;
        _error << file << ':' << line << std::endl;
        throw std::runtime_error(_error.str().c_str());
    }
}

#define checkCudaErrors(status) _checkCudaErrors(status, __FILE__, __LINE__);


inline const char *cublasGetErrorString(const cublasStatus_t& error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}


inline void _checkCublasErrors(const cublasStatus_t& status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::stringstream _error;
        _error << "Cublas failure: "
               << cublasGetErrorString(status)
               << std::endl;
        _error << file << ':' << line;
        throw std::runtime_error(_error.str().c_str());
    }
}

#define checkCublasErrors(status) _checkCublasErrors(status, __FILE__, __LINE__);


inline void InitializeCUDA(){
    int nDevices;

    cudaGetDeviceCount(&nDevices);

    if (nDevices < 1)
        throw std::runtime_error("Could not find GPU device");
    printf("Found %d devices:\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Number of concurrent kernels: %d\n",
               prop.concurrentKernels);
        printf("  Multi Processor Count: %d\n",
               prop.multiProcessorCount);
        printf("  Clock Frequency (KHz): %d\n",
               prop.clockRate);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total Global Memory (MB): %lu\n\n",
               prop.totalGlobalMem/1024/1024);
    }
    printf("Using GPU number 0\n");
    checkCudaErrors(cudaSetDevice(0));
    printf("------------------\n\n");
}



#endif //CUDNN_PROJ_HELPER_FUNCTIONS_CUH
