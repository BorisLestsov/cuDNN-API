#include "FullyConnectedLayer.cuh"


FullyConnectedLayer::FullyConnectedLayer(cublasHandle_t& cublas_handle_p,
                                         cudnnTensorDescriptor_t input_tensor_desc_p,
                                         size_t n_outputs_p):
        cublas_handle(cublas_handle_p),
        input_tensor_desc(input_tensor_desc_p),
        n_outp(n_outputs_p),
        _randrange(0.01f)
{
    int inp_strid;
    checkCudnnErrors( cudnnGetTensor4dDescriptor(input_tensor_desc,
                                                 &inp_datatype,
                                                 &in_N, &in_C, &in_H, &in_W,
                                                 &inp_strid, &inp_strid, &inp_strid, &inp_strid) );

    n_inp = in_C * in_H * in_W;
    out_C = 1;
    out_N = in_N;
    out_H = 1;
    out_W = n_outp;

    checkCudnnErrors( cudnnCreateTensorDescriptor(&output_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(output_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 inp_datatype,
                                                 out_N, out_C,
                                                 out_H, out_W) );

    h_weights = (float*) malloc(n_inp * n_outp * sizeof(float));
    h_bias = (float*) malloc(n_outp * sizeof(float));

    checkCudaErrors( cudaMalloc((void**) &d_weights, n_inp * n_outp * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &d_bias, n_outp * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &d_output, out_N * out_W * sizeof(float)) );

    checkCudaErrors( cudaMalloc((void**) &d_grad_w, n_inp * n_outp * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &d_grad_b, n_outp * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &d_dx, n_inp * in_N * sizeof(float)) );

    h_ones = (float*) malloc(out_W * in_N * sizeof(float));
    checkCudaErrors( cudaMalloc((void**) &d_ones, out_W * in_N *sizeof(float)) );
    std::fill_n(h_ones, out_W * in_N, 1.0f);
    checkCudaErrors( cudaMemcpy(d_ones, h_ones,
                                sizeof(float) * out_W * in_N, cudaMemcpyHostToDevice) );
}


FullyConnectedLayer::~FullyConnectedLayer() {
    free(h_weights);
    free(h_bias);
    free(h_ones);

    checkCudaErrors( cudaFree(d_weights) );
    checkCudaErrors( cudaFree(d_bias) );
    checkCudaErrors( cudaFree(d_output) );

    checkCudaErrors( cudaFree(d_grad_w) );
    checkCudaErrors( cudaFree(d_grad_b) );
    checkCudaErrors( cudaFree(d_dx) );

    checkCudaErrors( cudaFree(d_ones) );
}


void FullyConnectedLayer::init_weights_random(std::mt19937& gen){

    std::uniform_real_distribution<> get_rand(-_randrange, _randrange);

    size_t weights_length = n_inp * n_outp;
    size_t bias_length = n_outp;

    for (uint i = 0; i < weights_length; ++i)
        h_weights[i] = static_cast<float>(get_rand(gen));
    for (uint i = 0; i < bias_length; ++i)
        h_bias[i] = 1.0f;

    checkCudaErrors( cudaMemcpy(d_weights, h_weights,
                                sizeof(float) * weights_length, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_bias, h_bias,
                                sizeof(float) * bias_length, cudaMemcpyHostToDevice) );

}

void FullyConnectedLayer::propagate_forward(float* d_x) {
    float alpha = 1.0f;
    float beta = 0.0f;

//    float *h_x = (float *) malloc(std::max(out_N * out_C * out_H * out_W, in_N * in_C * in_H * in_W) * sizeof(float));
//    checkCudaErrors(cudaMemcpy(h_x, d_x,
//                               in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToHost));

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_outp, in_N, n_inp,
                                  &alpha,
                                  d_weights, n_inp,
                                  d_x, n_inp,
                                  &beta,
                                  d_output, n_outp));

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n_outp, in_N, 1,
                                  &alpha,
                                  d_bias, n_outp,
                                  d_ones, 1,
                                  &alpha,
                                  d_output, n_outp));

//    checkCudaErrors(cudaMemcpy(h_x, d_output,
//                               out_N * out_C * out_H * out_W * sizeof(float), cudaMemcpyDeviceToHost));
}


void FullyConnectedLayer::propagate_backward(float* d_dy, float* d_x) {
    float alpha = 1.0f;
    float beta = 0.0f;

    /*float *h_x = (float *) malloc(in_N * in_C * in_H * in_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_x, d_x,
                               in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToHost));
*/

    checkCublasErrors(cublasSgemm(cublas_handle,
                                  CUBLAS_OP_N, CUBLAS_OP_T,
                                  n_inp, n_outp, in_N,
                                  &alpha,
                                  d_x, n_inp,
                                  d_dy, n_outp,
                                  &beta,
                                  d_grad_w, n_inp));

    // Compute derivative with respect to bias: gfc2bias = dfc2smax * 1_vec
    checkCublasErrors(cublasSgemv(cublas_handle,
                                  CUBLAS_OP_N,
                                  n_outp, in_N,
                                  &alpha,
                                  d_dy, n_outp,
                                  d_ones, 1,
                                  &beta,
                                  d_grad_b, 1));

    // Compute derivative with respect to data (for previous layer): pfc2*dfc2smax (500x10*10xN)
    checkCublasErrors(cublasSgemm(cublas_handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  n_inp, in_N, n_outp,
                                  &alpha,
                                  d_weights, n_inp,
                                  d_dy, n_outp,
                                  &beta,
                                  d_dx, n_inp));


}

void FullyConnectedLayer::update_weights(float lr){
    float alpha = lr;

    int weights_length = n_inp * n_outp;
    int bias_length = n_outp;

    checkCublasErrors(cublasSaxpy(cublas_handle, weights_length,
                                  &alpha, d_grad_w, 1, d_weights, 1));
    checkCublasErrors(cublasSaxpy(cublas_handle, bias_length,
                                  &alpha, d_grad_b, 1, d_bias, 1));
}



/*

 Working Test

void FullyConnectedLayer::propagate_forward(float* d_x) {
    float alpha = 1.0f;
    float beta = 0.0f;



    n_inp = 2;
    n_outp = 3;

    out_H = 1;
    out_W = n_outp;

    in_N = 4;
    out_N = 4;



    uint i, j;

    for (i = 0; i < n_outp; ++i) {
        for (j = 0; j < n_inp; ++j) {
            h_weights[i * n_inp + j] = i * n_inp + j;
        }
    }
    for (i = 0; i < n_outp; ++i) {
        h_bias[i] = 1.0f;
    }


    for (i = 0; i < n_outp; ++i) {
        for (j = 0; j < n_inp; ++j) {
            std::cout << h_weights[i * n_inp + j] << "    ";
        }
        std::cout << std::endl;
    }

    float *h_x = (float *) malloc(sizeof(float) * n_inp * in_N);
    for (i = 0; i < in_N; ++i) {
        for (j = 0; j < n_inp; ++j) {
            h_x[i * n_inp + j] = 2 + i * n_inp + j;
        }
    }
    std::cout << "h_x:" << std::endl;
    for (i = 0; i < in_N; ++i) {
        for (j = 0; j < n_inp; ++j) {
            std::cout << h_x[i * n_inp + j] << "    ";
        }
        std::cout << std::endl;
    }
    checkCudaErrors(cudaMemcpy(d_x, h_x,
                               sizeof(float) * n_inp * in_N, cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMemcpy(d_weights, h_weights,
                               sizeof(float) * n_inp * n_outp, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bias, h_bias,
                               sizeof(float) * n_outp, cudaMemcpyHostToDevice));


    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  n_outp, in_N, n_inp,
                                  &alpha,
                                  d_weights, n_inp,
                                  d_x, n_inp,
                                  &beta,
                                  d_output, n_outp));

    checkCublasErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  n_outp, in_N, 1,
                                  &alpha,
                                  d_bias, n_outp,
                                  d_ones, 1,
                                  &alpha,
                                  d_output, n_outp));


    float *h_output = (float *) malloc(out_N * out_W * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_output, d_output,
                               out_N * out_W * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "res:" << std::endl;

    for (j = 0; j < out_N * out_W; ++j) {
        std::cout << h_output[j] << "    ";
    }
    std::cout << std::endl;


}
 */