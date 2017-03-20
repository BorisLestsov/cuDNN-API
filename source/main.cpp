#include "TrainData.cuh"
#include "ConvNet.cuh"

#include <iostream>

int main(){
    try {

        /*
         * TODO: add saving/loading weights !!!
         * TODO: add labels saving in TestData
         * TODO: add saving convolution filters - done
         * TODO: make possible train and test batch_sizes to be different
         */

        ulong seed = 2; // Should be passed through command line

        InitializeCUDA(0);

        cudnnHandle_t cudnn_handle;
        checkCudnnErrors(cudnnCreate(&cudnn_handle));
        cublasHandle_t cublas_handle;
        checkCublasErrors(cublasCreate(&cublas_handle));

        int train_batch_size = 256;
        int test_batch_size = train_batch_size;

        TrainData train(cudnn_handle,
                        "../dataset/imgdata.dat",
                        "../dataset/nmdata.dat",
                        "../dataset/lbldata.dat",
                        train_batch_size);

        TestData test(cudnn_handle,
                      "../dataset/imgdata.dat",
                      "../dataset/nmdata.dat",
                      train.n_labels,
                      test_batch_size);

        train.n_examples = 2560;


        ConvNet alexnet(cudnn_handle, cublas_handle, train.img_data_tensor_desc, seed);
        alexnet.fit(train, 10, 1e-5);
        //alexnet.predict(test);


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
