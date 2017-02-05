#include "TrainData.cuh"
#include "ConvNet.cuh"

#include <iostream>


int main(){
    InitializeCUDA();

    cudnnHandle_t cudnn_handle;
    checkCUDNN( cudnnCreate(&cudnn_handle) );
    cublasHandle_t cublas_handle;
    checkCudaErrors( cublasCreate(&cublas_handle) );

    TrainData train(cudnn_handle,
                    "dataset/imgdata.dat",
                    "dataset/nmdata.dat",
                    "dataset/lbldata.dat",
                    2);

    ConvNet alexnet(cudnn_handle, cublas_handle);
    alexnet.fit(train);


    checkCUDNN( cudnnDestroy(cudnn_handle) );
    checkCudaErrors( cublasDestroy(cublas_handle) );
    return 0;
}