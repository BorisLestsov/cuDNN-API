#include "SoftmaxLayer.cuh"

SoftmaxLayer::SoftmaxLayer(cudnnHandle_t& cudnn_handle_p,
                                 cudnnTensorDescriptor_t input_tensor_desc_p):
        cudnn_handle(cudnn_handle_p),
        input_tensor_desc(input_tensor_desc_p)
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

    n_labels = out_C * out_H * out_W;

    checkCudnnErrors( cudnnCreateTensorDescriptor(&output_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(output_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 inp_datatype,
                                                 out_N, out_C,
                                                 out_H, out_W) );

    checkCudaErrors( cudaMalloc(&d_output, sizeof(float) * out_N * out_C * out_H * out_W) );
    checkCudaErrors( cudaMalloc(&d_dx, sizeof(float) * in_N * in_C * in_H * in_W) );

}

SoftmaxLayer::~SoftmaxLayer() {
    cudnnDestroyTensorDescriptor(output_tensor_desc);

    checkCudaErrors( cudaFree(d_output) );
    checkCudaErrors( cudaFree(d_dx) );
}


void SoftmaxLayer::propagate_forward(float* d_x){
    float alpha = 1.0f, beta = 0.0f;

    checkCudnnErrors( cudnnSoftmaxForward(cudnn_handle,
                                          CUDNN_SOFTMAX_ACCURATE,
                                          CUDNN_SOFTMAX_MODE_INSTANCE,
                                          &alpha,
                                          input_tensor_desc, d_x,
                                          &beta,
                                          output_tensor_desc, d_output) );

//    float *h_output = (float *) malloc(out_N * out_W * sizeof(float));
//    checkCudaErrors(cudaMemcpy(h_output, d_output,
//                               out_N * out_C * out_H * out_W * sizeof(float), cudaMemcpyDeviceToHost));
//    std::cout << "Softmax:" << std::endl;
//
//    for (uint i = 0; i < out_N; ++i) {
//        std::cout << "    EXAMPLE" << std::endl;
//        for (uint j = 0; j < out_W; ++j) {
//            std::cout << h_output[i*out_W + j] << "    ";
//        }
//        std::cout << std::endl;
//    }
}


void SoftmaxLayer::propagate_backward(float* d_dy, float* d_x){
    float alpha = 1.0f, beta = 0.0f;

    /*float *h_x = (float *) malloc(in_N * in_C * in_H * in_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_x, d_x,
                               in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToHost));
*/


    checkCudnnErrors( cudnnSoftmaxBackward(cudnn_handle,
                                           CUDNN_SOFTMAX_ACCURATE,
                                           CUDNN_SOFTMAX_MODE_INSTANCE,
                                           &alpha,
                                           output_tensor_desc, d_output,
                                           output_tensor_desc, d_dy,
                                           &beta,
                                           input_tensor_desc, d_dx) );

}

