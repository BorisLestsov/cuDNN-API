#include "TrainData.cuh"
#include "ConvNet.cuh"

#include <iostream>

int main(){
    try {

        /*
         * TODO: add saving/loading weights
         * TODO: implement TestData
         * TODO: add saving convolution filters
         */

        ulong seed = 1; // Should be passed through command line

        InitializeCUDA();

        cudnnHandle_t cudnn_handle;
        checkCudnnErrors(cudnnCreate(&cudnn_handle));
        cublasHandle_t cublas_handle;
        checkCublasErrors(cublasCreate(&cublas_handle));

        TrainData train(cudnn_handle,
                        "dataset/imgdata.dat",
                        "dataset/nmdata.dat",
                        "dataset/lbldata.dat",
                        1);


        ConvNet alexnet(cudnn_handle, cublas_handle, train.img_data_tensor_desc, seed);
        alexnet.fit(train, 500, 1e-2);

        //alexnet.conv1.save_kernels("kernels.dat");


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