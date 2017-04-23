#include "LayerFactory.hpp"

LayerFactory::LayerFactory(cudnnHandle_t& cudnn_handle_p,
             cublasHandle_t& cublas_handle_p,
             size_t seed)
        :
        cudnn_handle(cudnn_handle_p),
        cublas_handle(cublas_handle_p),
        gen(seed == 0 ? rd() : seed)
{}

Layer* LayerFactory::CreateConvolutionLayer(cudnnTensorDescriptor_t input_tensor_desc_p,
                                           size_t depth, size_t ker_size, size_t stride, size_t zp)
{
    ConvolutionLayer* layer = new ConvolutionLayer(cudnn_handle,
                                                   cublas_handle,
                                                   input_tensor_desc_p,
                                                   depth,
                                                   ker_size,
                                                   stride,
                                                   zp);
    layer->init_weights_random(gen);
    return layer;
}

Layer* LayerFactory::CreateActivationLayer(cudnnTensorDescriptor_t input_tensor_desc_p,
                                           cudnnActivationMode_t act_f_p)
{
    ActivationLayer* layer = new ActivationLayer(cudnn_handle, input_tensor_desc_p, act_f_p);
    return layer;
}

Layer* LayerFactory::CreateFullyConnectedLayer(cudnnTensorDescriptor_t input_tensor_desc_p, size_t n_outputs_p)
{
    FullyConnectedLayer* layer = new FullyConnectedLayer(cublas_handle, input_tensor_desc_p, n_outputs_p);
    layer->init_weights_random(gen);
    return layer;
}

Layer* LayerFactory::CreatePoolingLayer(cudnnTensorDescriptor_t input_tensor_desc_p,
                                        size_t size_p, size_t stride_p, size_t pad_p)
{
    PoolingLayer* layer = new PoolingLayer(cudnn_handle, input_tensor_desc_p, size_p, stride_p, pad_p);
    return layer;
}

Layer* LayerFactory::CreateSoftmaxLayer(cudnnTensorDescriptor_t input_tensor_desc_p)
{
    SoftmaxLayer* layer = new SoftmaxLayer(cudnn_handle, input_tensor_desc_p);
    return layer;
}

MetricLayer* LayerFactory::CreateNLLMetric(cudnnTensorDescriptor_t input_tensor_desc_p)
{
    NegLogLikelihoodLayer* layer = new NegLogLikelihoodLayer(cudnn_handle, input_tensor_desc_p);
    return layer;
}

MetricLayer* LayerFactory::CreateMSEMetric(cudnnTensorDescriptor_t input_tensor_desc_p)
{
    MSELayer* layer = new MSELayer(cudnn_handle, input_tensor_desc_p);
    return layer;
}