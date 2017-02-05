#include "FullyConnectedLayer.cuh"


FullyConnectedLayer::FullyConnectedLayer(cublasHandle_t& cublas_handle_p, size_t n_inp_p, size_t n_outp_p):
        cublas_handle(cublas_handle_p),
        n_inp(n_inp_p),
        n_outp(n_outp_p)
{
    weights = (float*) malloc(n_inp * n_outp);
    bias = (float*) malloc(n_outp);

    checkCudaErrors( cudaMalloc(&d_weights, n_inp * n_outp * sizeof(float)) );
    checkCudaErrors( cudaMalloc(&d_bias, n_outp * sizeof(float)) );
}


FullyConnectedLayer::~FullyConnectedLayer() {
    free(weights);
    free(bias);

    checkCudaErrors( cudaFree(d_weights) );
    checkCudaErrors( cudaFree(d_bias) );
}


void FullyConnectedLayer::init_weights_random(/* rand function?*/){

    std::fill(weights, weights + n_inp*n_outp, 2.0f);
    std::fill(bias, bias + n_outp, 2.0f);

    checkCudaErrors( cudaMemcpy(d_weights, weights,
                                     sizeof(float) * n_inp * n_outp, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_bias, bias,
                                     sizeof(float) * n_outp, cudaMemcpyHostToDevice) );

}

void FullyConnectedLayer::propagate_forward(/* ... */){
    float alpha = 1.0f;
    float beta = 0.0f;

    checkCudaErrors( cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 2,2,2,
                                 &alpha,
                                 d_weights, 2,
                                 d_weights, 2,
                                 0,
                                 d_weights, 2) );

    checkCudaErrors( cudaMemcpyAsync(weights, d_weights,
                                     sizeof(float) * n_inp * n_outp, cudaMemcpyDeviceToHost) );

    for (uint i = 0; i < 2; ++i){
        for (uint j = 0; j < 2; ++j){
            std::cout << weights[i*2 + j]  << "    ";
        }
        std::cout << std::endl;
    }
}