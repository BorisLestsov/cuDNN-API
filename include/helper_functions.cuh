#ifndef CUDNN_PROJ_HELPER_FUNCTIONS_CUH
#define CUDNN_PROJ_HELPER_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <cudnn.h>

#include <sstream>
#include <iostream>
#include <cstdlib>


inline void init_cuda(){
    int nDevices;

    cudaGetDeviceCount(&nDevices);

    if (nDevices < 1)
        throw std::runtime_error("Could not find GPU device");

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Number of concurrent kernels: %d\n",
               prop.concurrentKernels);
        printf("  Multi Processor Count: %d\n",
               prop.multiProcessorCount);
        printf("  Total Global Memory (MB): %lu\n",
               prop.totalGlobalMem/1024/1024);
    }
    printf("Using GPU number 0\n");
    printf("------------------\n\n");
}


#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)


#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while (0)


#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while (0)



#endif //CUDNN_PROJ_HELPER_FUNCTIONS_CUH
