#include "Layer.cuh"

Layer::Layer(Layer_t type_p, cudnnTensorDescriptor_t input_tensor_desc,
             cudnnHandle_t cudnn_handle,
             cublasHandle_t cublas_handle)
        :
        type(type_p),
        input_tensor_desc(input_tensor_desc),
        cublas_handle(cublas_handle),
        cudnn_handle(cudnn_handle)
{
    int inp_strid;
    checkCudnnErrors( cudnnGetTensor4dDescriptor(input_tensor_desc,
                                                 &inp_datatype,
                                                 &in_N, &in_C, &in_H, &in_W,
                                                 &inp_strid, &inp_strid, &inp_strid, &inp_strid) );
}


MetricLayer::MetricLayer(Layer_t type_p,
                         cudnnTensorDescriptor_t input_tensor_desc,
                         cudnnHandle_t cudnn_handle,
                         cublasHandle_t cublas_handle):
        Layer(type_p, input_tensor_desc, cudnn_handle, cublas_handle)
{}