#include "TrainData.cuh"
#include "ConvNet.hpp"

#include "LayerFactory.hpp"

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

        size_t train_batch_size = 25;
        size_t test_batch_size = train_batch_size;

        TrainData train(cudnn_handle,
                        "./dataset/imgdata.dat",
                        "./dataset/nmdata.dat",
                        "./dataset/lbldata.dat",
                        train_batch_size);

        TestData test(cudnn_handle,
                      "./dataset/imgdata.dat",
                      "./dataset/nmdata.dat",
                      train.n_labels,
                      test_batch_size);

	train.n_examples = 1000;

        ConvNet alexnet(cudnn_handle, cublas_handle, train.img_data_tensor_desc);
        LayerFactory lf(cudnn_handle, cublas_handle, seed);

        // Layers creation

        alexnet.add_layer(lf.CreateConvolutionLayer(train.img_data_tensor_desc,
                                                    96, 11, 4, 0));
        alexnet.add_layer(lf.CreateActivationLayer(alexnet.last_layer_outp_desc(),
                                                   CUDNN_ACTIVATION_RELU));
        alexnet.add_layer(lf.CreatePoolingLayer(alexnet.last_layer_outp_desc(),
                                                3, 2, 0));


        alexnet.add_layer(lf.CreateConvolutionLayer(alexnet.last_layer_outp_desc(),
                                                    256, 5, 1, 1));
        alexnet.add_layer(lf.CreateActivationLayer(alexnet.last_layer_outp_desc(),
                                                   CUDNN_ACTIVATION_RELU));
        alexnet.add_layer(lf.CreatePoolingLayer(alexnet.last_layer_outp_desc(),
                                                3, 2, 1));


        alexnet.add_layer(lf.CreateConvolutionLayer(alexnet.last_layer_outp_desc(),
                                                    384, 3, 1, 1));
        alexnet.add_layer(lf.CreateActivationLayer(alexnet.last_layer_outp_desc(),
                                                   CUDNN_ACTIVATION_RELU));


        alexnet.add_layer(lf.CreateConvolutionLayer(alexnet.last_layer_outp_desc(),
                                                    384, 3, 1, 1));
        alexnet.add_layer(lf.CreateActivationLayer(alexnet.last_layer_outp_desc(),
                                                   CUDNN_ACTIVATION_RELU));


        alexnet.add_layer(lf.CreateConvolutionLayer(alexnet.last_layer_outp_desc(),
                                                    256, 3, 1, 1));
        alexnet.add_layer(lf.CreateActivationLayer(alexnet.last_layer_outp_desc(),
                                                   CUDNN_ACTIVATION_RELU));
        alexnet.add_layer(lf.CreatePoolingLayer(alexnet.last_layer_outp_desc(),
                                                3, 2, 1));


        alexnet.add_layer(lf.CreateFullyConnectedLayer(alexnet.last_layer_outp_desc(),
                                                       1000));
        alexnet.add_layer(lf.CreateActivationLayer(alexnet.last_layer_outp_desc(),
                                                   CUDNN_ACTIVATION_RELU));


        alexnet.add_layer(lf.CreateFullyConnectedLayer(alexnet.last_layer_outp_desc(),
                                                       1000));
        alexnet.add_layer(lf.CreateActivationLayer(alexnet.last_layer_outp_desc(),
                                                   CUDNN_ACTIVATION_RELU));


        alexnet.add_layer(lf.CreateFullyConnectedLayer(alexnet.last_layer_outp_desc(),
                                                       91));
        alexnet.add_layer(lf.CreateSoftmaxLayer(alexnet.last_layer_outp_desc()));


        alexnet.set_metric(lf.CreateNLLMetric(alexnet.last_layer_outp_desc()));


        //Training

        alexnet.fit(train, 500, 1e-4, 0.98);
/*        auto res = alexnet.predict_labels(test);
        for (auto i: res){
            std::cout << i << std::endl;
        }
*/

        checkCudnnErrors(cudnnDestroy(cudnn_handle));
        checkCublasErrors(cublasDestroy(cublas_handle));
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl << "Aborting..." << std::endl;
        cudaDeviceReset();
        return 1;
    }
    catch (...) {
        std::cerr <<"Unknown exception" << std::endl << "Aborting..." << std::endl;
        cudaDeviceReset();
        return 1;
    }
    return 0;
}
