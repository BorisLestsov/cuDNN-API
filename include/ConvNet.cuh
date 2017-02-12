#ifndef CUDNN_PROJ_CONVNET_CUH
#define CUDNN_PROJ_CONVNET_CUH

#include "helper_functions.cuh"
#include "TrainData.cuh"
#include "TestData.h"
#include "ConvolutionLayer.cuh"
#include "PoolingLayer.cuh"
#include "FullyConnectedLayer.cuh"

#include <random>

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>


class ConvNet {
public:

    ConvNet(cudnnHandle_t& cudnn_handle,
            cublasHandle_t& cublas_handle,
            cudnnTensorDescriptor_t data_tensor_desc_p,
            uint seed = 0);

    void fit(TrainData&);
    char* predict(TestData&);

private:

    ConvolutionLayer conv1;
    PoolingLayer pool1;
    FullyConnectedLayer fc1;

    cudnnHandle_t& cudnn_handle;
    cublasHandle_t& cublas_handle;
    cudnnTensorDescriptor_t data_tensor_desc;

    std::random_device rd;
    std::mt19937 gen;
};


#endif //CUDNN_PROJ_CONVNET_CUH
