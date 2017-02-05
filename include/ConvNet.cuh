#ifndef CUDNN_PROJ_CONVNET_CUH
#define CUDNN_PROJ_CONVNET_CUH

#include "helper_functions.cuh"
#include "TrainData.cuh"
#include "TestData.h"
#include "FullyConnectedLayer.cuh"

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>


class ConvNet {
public:

    ConvNet(cudnnHandle_t& cudnn_handle, cublasHandle_t& cublas_handle);

    void fit(TrainData&);
    char* predict(TestData&);

private:

    FullyConnectedLayer fc1;

    cudnnHandle_t& cudnn_handle;
    cublasHandle_t& cublas_handle;

};


#endif //CUDNN_PROJ_CONVNET_CUH
