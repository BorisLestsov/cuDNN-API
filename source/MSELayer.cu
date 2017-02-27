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
    out_C = in_C;
    out_H = in_H;
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
    float alpha = 1.0f, beta = 0.0f;

//    float *h_x = (float *) malloc(in_N * in_C * in_H * in_W * sizeof(float));
//    checkCudaErrors(cudaMemcpy(h_x, d_x,
//                               in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToHost));


    int BW = 128;
    compute_mse<<<_ceil(in_N, BW), BW>>>(d_t, d_x, n_labels, in_N, d_output);



    float *h_output = (float *) malloc(out_N * out_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_output, d_output,
                               out_N * out_C * out_H * out_W * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "MSE:" << std::endl;

    for (uint i = 0; i < out_N; ++i) {
        std::cout << "    Batch:" << h_output[i] << std::endl;
    }
}


void MSELayer::propagate_backward(float* d_targ, float* d_dx){
    float alpha = 1.0f, beta = 0.0f;

    /*float *h_x = (float *) malloc(in_N * in_C * in_H * in_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_x, d_x,
                               in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToHost));


    int BW = 128;

    checkCudaErrors(cudaMemcpyAsync(d_dx,
                                    d_output,
                                    sizeof(float) * out_N * n_labels, cudaMemcpyDeviceToDevice));

    //compute_mse<<<_ceil(out_N, BW), BW>>>(d_targ, n_labels, out_N, d_dx);



    /*float *h_output = (float *) malloc(out_N * out_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_output, d_output,
                               out_N * out_C * out_H * out_W * sizeof(float), cudaMemcpyDeviceToHost));
    */
}


__global__ void compute_mse(const float *labels, const float* x, int num_labels, int batch_size, float* losses){
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
