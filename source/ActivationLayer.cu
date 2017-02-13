#include "ActivationLayer.cuh"

ActivationLayer::ActivationLayer(cudnnHandle_t& cudnn_handle_p,
                                 cudnnTensorDescriptor_t input_tensor_desc_p,
                                 cudnnActivationMode_t act_f_p):
        cudnn_handle(cudnn_handle_p),
        input_tensor_desc(input_tensor_desc_p),
        act_f(act_f_p)
{
    int inp_strid;
    checkCudnnErrors( cudnnGetTensor4dDescriptor(input_tensor_desc,
                                                 &inp_datatype,
                                                 &in_N, &in_C, &in_H, &in_W,
                                                 &inp_strid, &inp_strid, &inp_strid, &inp_strid) );

    out_N = in_N;
    out_C = in_C;
    out_H = in_H;
    out_W = in_W;

    checkCudnnErrors( cudnnCreateActivationDescriptor(&act_desc) );
    checkCudnnErrors( cudnnSetActivationDescriptor(act_desc,
                                                   act_f,
                                                   CUDNN_PROPAGATE_NAN,
                                                   0.0f) );     // TODO: Add clipped relu

    checkCudnnErrors( cudnnCreateTensorDescriptor(&output_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(output_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 inp_datatype,
                                                 out_N, out_C,
                                                 out_H, out_W) );

    checkCudaErrors( cudaMalloc(&d_output, sizeof(float) * out_N * out_C * out_H * out_W) );

}

ActivationLayer::~ActivationLayer() {
    cudnnDestroyActivationDescriptor(act_desc);
    cudnnDestroyTensorDescriptor(output_tensor_desc);

    checkCudaErrors( cudaFree(d_output) );
}


void ActivationLayer::propagate_forward(float* d_x){
    float alpha = 1.0f, beta = 0.0f;

    checkCudnnErrors( cudnnActivationForward(cudnn_handle,
                                             act_desc,
                                             &alpha,
                                             input_tensor_desc,
                                             d_x,
                                             &beta,
                                             output_tensor_desc,
                                             d_output) );

}

