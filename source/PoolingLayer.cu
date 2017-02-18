#include "PoolingLayer.cuh"

PoolingLayer::PoolingLayer(cudnnHandle_t& cudnn_handle_p,
                           cudnnTensorDescriptor_t input_tensor_desc_p,
                           size_t size_p,
                           size_t stride_p):
        cudnn_handle(cudnn_handle_p),
        input_tensor_desc(input_tensor_desc_p),
        size(size_p),
        stride(stride_p)
{
    int inp_strid;
    checkCudnnErrors( cudnnGetTensor4dDescriptor(input_tensor_desc,
                                                 &inp_datatype,
                                                 &in_N, &in_C, &in_H, &in_W,
                                                 &inp_strid, &inp_strid, &inp_strid, &inp_strid) );

    checkCudnnErrors( cudnnCreatePoolingDescriptor(&pooling_desc) );
    checkCudnnErrors( cudnnSetPooling2dDescriptor(pooling_desc,
                                                  CUDNN_POOLING_MAX,
                                                  CUDNN_PROPAGATE_NAN,
                                                  size, size,
                                                  0, 0,
                                                  stride, stride) );
    checkCudnnErrors( cudnnGetPooling2dForwardOutputDim(pooling_desc,
                                                        input_tensor_desc,
                                                        &out_N, &out_C, &out_H, &out_W) );

    checkCudnnErrors( cudnnCreateTensorDescriptor(&output_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(output_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 inp_datatype,
                                                 out_N, out_C,
                                                 out_H, out_W) );

    checkCudaErrors( cudaMalloc(&d_output, sizeof(float) * out_N * out_C * out_H * out_W) );

}

PoolingLayer::~PoolingLayer() {
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroyTensorDescriptor(output_tensor_desc);

    checkCudaErrors( cudaFree(d_output) );
}


void PoolingLayer::propagate_forward(float* d_x){
    float alpha = 1.0f, beta = 0.0f;

    float *h_x = (float *) malloc(in_N * in_C * in_H * in_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_x, d_x,
                               in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudnnErrors( cudnnPoolingForward(cudnn_handle,
                                          pooling_desc,
                                          &alpha,
                                          input_tensor_desc, d_x,
                                          &beta,
                                          output_tensor_desc, d_output) );
}

