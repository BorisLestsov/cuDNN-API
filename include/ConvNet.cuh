#ifndef CUDNN_PROJ_CONVNET_CUH
#define CUDNN_PROJ_CONVNET_CUH

#include "helper_functions.cuh"
#include "TrainData.cuh"
#include "TestData.h"

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>


class ConvNet {
public:

    cudnnHandle_t& cudnn_handle;
    cublasHandle_t& cublas_handle;

    ConvNet(cudnnHandle_t& cudnn_handle, cublasHandle_t& cublas_handle);


    void fit(TrainData&);
    char* predict(TestData&);

private:
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
};


#endif //CUDNN_PROJ_CONVNET_CUH
