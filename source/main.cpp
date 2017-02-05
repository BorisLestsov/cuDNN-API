#include "TrainData.cuh"
#include "ConvNet.cuh"

#include <iostream>

#define SEED 1

int main(){
    try {
        InitializeCUDA();

        cudnnHandle_t cudnn_handle;
        checkCUDNN(cudnnCreate(&cudnn_handle));
        cublasHandle_t cublas_handle;
        checkCudaErrors(cublasCreate(&cublas_handle));

        TrainData train(cudnn_handle,
                        "dataset/imgdata.dat",
                        "dataset/nmdata.dat",
                        "dataset/lbldata.dat",
                        2);

        ConvNet alexnet(cudnn_handle, cublas_handle);
        alexnet.fit(train);


        checkCUDNN(cudnnDestroy(cudnn_handle));
        checkCudaErrors(cublasDestroy(cublas_handle));
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}