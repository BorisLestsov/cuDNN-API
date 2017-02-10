#include "TrainData.cuh"
#include "ConvNet.cuh"

#include <iostream>

int main(){
    try {

        ulong seed = 0; // Should be passed through command line

        InitializeCUDA();

        cudnnHandle_t cudnn_handle;
        checkCudnnErrors(cudnnCreate(&cudnn_handle));
        cublasHandle_t cublas_handle;
        checkCublasErrors(cublasCreate(&cublas_handle));

        TrainData train(cudnn_handle,
                        "dataset/imgdata.dat",
                        "dataset/nmdata.dat",
                        "dataset/lbldata.dat",
                        2);


        ConvNet alexnet(cudnn_handle, cublas_handle, train.img_data_tensor_desc);
        alexnet.fit(train);


        checkCudnnErrors(cudnnDestroy(cudnn_handle));
        checkCublasErrors(cublasDestroy(cublas_handle));
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl << "Aborting..." << std::endl;
        cudaDeviceReset();
        return 1;
    }
    return 0;
}