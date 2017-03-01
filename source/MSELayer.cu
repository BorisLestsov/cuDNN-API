#include "MSELayer.cuh"

MSELayer::MSELayer(cudnnHandle_t& cudnn_handle_p,
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
    out_C = 1;
    out_H = 1;
    out_W = 1;

    n_labels = in_C * in_H * in_W;

    checkCudnnErrors( cudnnCreateTensorDescriptor(&output_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(output_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 inp_datatype,
                                                 out_N, out_C,
                                                 out_H, out_W) );

    checkCudaErrors( cudaMalloc(&d_output, sizeof(float) * out_N * out_C * out_H * out_W) );
    checkCudaErrors( cudaMalloc(&d_dx, sizeof(float) * in_N * in_C * in_H * in_W) );

}

MSELayer::~MSELayer() {
    cudnnDestroyTensorDescriptor(output_tensor_desc);

    checkCudaErrors( cudaFree(d_output) );
    checkCudaErrors( cudaFree(d_dx) );
}


void MSELayer::propagate_forward(float* d_t, float* d_x){

    compute_mse<<<_ceil(in_N, BW), BW>>>(d_t, d_x, n_labels, in_N, d_output);



    float *h_output = (float *) malloc(out_N * out_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_output, d_output,
                               out_N * out_C * out_H * out_W * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "MSE:" << std::endl;

    float batch_loss = 0.0;
    for (uint i = 0; i < out_N; ++i) {
        batch_loss += h_output[i];
    }

    std::cout << "    Batch loss:" << batch_loss << std::endl;
}


void MSELayer::propagate_backward(float* d_t, float* d_y){

    compute_mse_loss<<<_ceil(out_N, BW), BW>>>(d_t, d_y, n_labels, in_N, d_dx);



//    float *h_output = (float *) malloc(in_N * in_C * in_H * in_W * sizeof(float));
//    checkCudaErrors(cudaMemcpy(h_output, d_dx,
//                               in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToHost));
//    std::cout << "MSE:" << std::endl;
//
//    for (uint i = 0; i < in_N; ++i) {
//        std::cout << "    Batch gradient:" << std::endl;
//        for (uint j = 0; j < in_C*in_H*in_W; ++j) {
//            std::cout << h_output[i*n_labels + j] << "    ";
//        }
//        std::cout << std::endl;
//    }
}


__global__ void compute_mse(const float *labels,
                            const float* x,
                            int num_labels,
                            int batch_size,
                            float* losses)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size)
        return;


    int i, stride = batch_idx * num_labels;
    float lbl_diff, loss = 0.0f;
    for (i = 0; i < num_labels; ++i){
        lbl_diff = labels[stride + i] - x[stride + i];
        loss += lbl_diff * lbl_diff;
    }
    losses[batch_idx] = loss;
}


__global__ void compute_mse_loss(const float *labels,
                                 const float* y,
                                 int num_labels,
                                 int batch_size,
                                 float* grad)
{
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size)
        return;


    int i, stride = batch_idx * num_labels;
    for (i = 0; i < num_labels; ++i){
        grad[stride + i] = labels[stride + i] - y[stride + i];
    }
}