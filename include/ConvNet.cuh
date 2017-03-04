#ifndef CUDNN_PROJ_CONVNET_CUH
#define CUDNN_PROJ_CONVNET_CUH

#include "helper_functions.cuh"
#include "TrainData.cuh"
#include "TestData.cuh"

#include "ConvolutionLayer.cuh"
#include "PoolingLayer.cuh"
#include "FullyConnectedLayer.cuh"
#include "ActivationLayer.cuh"
#include "SoftmaxLayer.cuh"
#include "MSELayer.cuh"
#include "NegLogLikelihoodLayer.cuh"

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

    void fit(TrainData& data, int epoches, float lr);
    int* predict(TestData& data);


private:


    ConvolutionLayer conv1;
    FullyConnectedLayer fc1;
    ActivationLayer act1;
    FullyConnectedLayer fc2;
    SoftmaxLayer sm;
    NegLogLikelihoodLayer nll;


    cudnnHandle_t& cudnn_handle;
    cublasHandle_t& cublas_handle;
    cudnnTensorDescriptor_t data_tensor_desc;

    std::random_device rd;
    std::mt19937 gen;
};


#endif //CUDNN_PROJ_CONVNET_CUH
