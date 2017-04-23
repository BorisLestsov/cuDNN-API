#ifndef CUDNN_PROJ_LAYERFACTORY_H
#define CUDNN_PROJ_LAYERFACTORY_H

#include "Layer.cuh"

#include "ConvolutionLayer.cuh"
#include "PoolingLayer.cuh"
#include "FullyConnectedLayer.cuh"
#include "ActivationLayer.cuh"
#include "SoftmaxLayer.cuh"
#include "MSELayer.cuh"
#include "NegLogLikelihoodLayer.cuh"


class LayerFactory {
public:

    LayerFactory(cudnnHandle_t& cudnn_handle_p,
                 cublasHandle_t& cublas_handle_p,
                 size_t seed);

    Layer* CreateConvolutionLayer(cudnnTensorDescriptor_t input_tensor_desc_p,
                                  size_t depth, size_t ker_size, size_t stride, size_t zp = 0);

    Layer* CreateActivationLayer(cudnnTensorDescriptor_t input_tensor_desc_p,
                                 cudnnActivationMode_t act_f_p);

    Layer* CreateFullyConnectedLayer(cudnnTensorDescriptor_t input_tensor_desc_p, size_t n_outputs_p);

    Layer* CreatePoolingLayer(cudnnTensorDescriptor_t input_tensor_desc_p,
                              size_t size_p, size_t stride_p, size_t pad_p);

    Layer* CreateSoftmaxLayer(cudnnTensorDescriptor_t input_tensor_desc_p);

    MetricLayer* CreateNLLMetric(cudnnTensorDescriptor_t input_tensor_desc_p);

    MetricLayer* CreateMSEMetric(cudnnTensorDescriptor_t input_tensor_desc_p);


private:
    cudnnHandle_t& cudnn_handle;
    cublasHandle_t& cublas_handle;

    std::random_device rd;
    std::mt19937 gen;
};


#endif //CUDNN_PROJ_LAYERFACTORY_H
