#include "FullyConnectedLayer.cuh"


FullyConnectedLayer::FullyConnectedLayer(cublasHandle_t& cublas_handle_p, size_t n_inp_p, size_t n_outp_p):
        cublas_handle(cublas_handle_p),
        n_inp(n_inp_p),
        n_outp(n_outp_p)
{
    h_weights = (float*) malloc(n_inp * n_outp * sizeof(float));
    h_bias = (float*) malloc(n_outp * sizeof(float));

    checkCudaErrors( cudaMalloc((void**) &d_weights, n_inp * n_outp * sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**) &d_bias, n_outp * sizeof(float)) );
}


FullyConnectedLayer::~FullyConnectedLayer() {
    free(h_weights);
    free(h_bias);

    checkCudaErrors( cudaFree(d_weights) );
    checkCudaErrors( cudaFree(d_bias) );
}


void FullyConnectedLayer::init_weights_random(/* rand function?*/){

    uint i, j;

    for (i = 0; i < n_outp; ++i){
        for (j = 0; j < n_inp; ++j){
            h_weights[i*n_inp + j] = i*n_inp + j;
        }
    }


    for (i = 0; i < n_outp; ++i){
        for (j = 0; j < n_inp; ++j){
            std::cout << h_weights[i*n_inp + j]  << "    ";
        }
        std::cout << std::endl;
    }

    checkCudaErrors( cudaMemcpy(d_weights, h_weights,
                                     sizeof(float) * n_inp * n_outp, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_bias, h_bias,
                                     sizeof(float) * n_outp, cudaMemcpyHostToDevice) );

}

void FullyConnectedLayer::propagate_forward(float* d_x){
    float alpha = 1.0f;
    float beta = 0.0f;


    /*
     * THis is a working example of Sgevm:

     n_inp = true cols
     n_outp = true rows
    checkCublasErrors( cublasSgemv(cublas_handle,
                                   CUBLAS_OP_T,
                                   n_inp, n_outp,
                                   &alpha,
                                   d_weights, n_inp,
                                   d_x, 1,
                                   &beta,
                                   d_x, 1) );
    */

    /*

    Working gemm

    checkCublasErrors( cublasSgemm(cublas_handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   colsB, rowsA, colsA
                                   &alpha,
                                   d_B, colsB,
                                   d_A, colsA,
                                   &beta,
                                   d_Store, colsStore) );
    */

}