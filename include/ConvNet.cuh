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


    ConvolutionLayer conv1;
    ActivationLayer relu1;
    PoolingLayer pool1;
    ConvolutionLayer conv2;
    ActivationLayer relu2;
    PoolingLayer pool2;
    ConvolutionLayer conv3;
    ActivationLayer relu3;
    ConvolutionLayer conv4;
    ActivationLayer relu4;
    ConvolutionLayer conv5;
    ActivationLayer relu5;
    PoolingLayer pool5;

    FullyConnectedLayer fc6;
    ActivationLayer relu6;
    //drop 6
    FullyConnectedLayer fc7;
    ActivationLayer relu7;
    //drop 7
    FullyConnectedLayer fc8;
    SoftmaxLayer sm;
    NegLogLikelihoodLayer nll;

private:

    cudnnHandle_t& cudnn_handle;
    cublasHandle_t& cublas_handle;
    cudnnTensorDescriptor_t data_tensor_desc;

    std::random_device rd;
    std::mt19937 gen;
};


#endif //CUDNN_PROJ_CONVNET_CUH
