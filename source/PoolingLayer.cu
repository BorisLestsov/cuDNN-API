#include "PoolingLayer.cuh"

PoolingLayer::PoolingLayer(cudnnHandle_t& cudnn_handle_p,
                           cudnnTensorDescriptor_t input_tensor_desc_p,
                           size_t size_p,
                           size_t stride_p, 
			   size_t pad_p):

        Layer(Layer_t::Pooling, input_tensor_desc_p, cudnn_handle_p, nullptr),
        size(size_p),
        stride(stride_p),
	    pad(pad_p)
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
                                                  pad, pad,
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
    checkCudaErrors( cudaMalloc(&d_dx, sizeof(float) * in_N * in_C * in_H * in_W) );
    
    std::cout << "pool in:  " << in_N << " " << in_C << " " << in_H << " " << in_W << std::endl;
    std::cout << "pool out: " << out_N << " " << out_C << " " << out_H << " " << out_W << std::endl;

}

PoolingLayer::~PoolingLayer() {
    cudnnDestroyPoolingDescriptor(pooling_desc);
    cudnnDestroyTensorDescriptor(output_tensor_desc);

    checkCudaErrors( cudaFree(d_output) );
    checkCudaErrors( cudaFree(d_dx) );
}


void PoolingLayer::propagate_forward(float* d_x){
    float alpha = 1.0f, beta = 0.0f;

#ifdef DEBUG    
    std::cout << "pool in: " << cudaCheckNan(d_x, in_N*in_C*in_H*in_W) << std::endl;
#endif

    checkCudnnErrors( cudnnPoolingForward(cudnn_handle,
                                          pooling_desc,
                                          &alpha,
                                          input_tensor_desc, d_x,
                                          &beta,
                                          output_tensor_desc, d_output) );

#ifdef DEBUG
    std::cout << "pool out: " << cudaCheckNan(d_output, out_N*out_C*out_H*out_W) << std::endl;
#endif
}


void PoolingLayer::propagate_backward(float* d_dy, float* d_x, float momentum){
    float alpha = 1.0f, beta = 0.0f;

#ifdef DEBUG    
    std::cout << "back pool in: " << cudaCheckNan(d_dy, out_N*out_C*out_H*out_W) << std::endl;
#endif

    checkCudnnErrors(cudnnPoolingBackward(cudnn_handle,
                                          pooling_desc,
                                          &alpha,
                                          output_tensor_desc, d_output,
                                          output_tensor_desc, d_dy,
                                          input_tensor_desc, d_x,
                                          &beta,
                                          input_tensor_desc, d_dx));

#ifdef DEBUG
    std::cout << " back pool out: " << cudaCheckNan(d_dx, in_N*in_C*in_H*in_W) << std::endl;
#endif
}
