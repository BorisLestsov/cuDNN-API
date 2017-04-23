#ifndef CUDNN_PROJ_CONVNET_CUH
#define CUDNN_PROJ_CONVNET_CUH

#include "helper_functions.cuh"
#include "TrainData.cuh"
#include "TestData.cuh"
#include "Layer.cuh"

#include <random>

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>


class ConvNet {
public:

    ConvNet(cudnnHandle_t& cudnn_handle,
            cublasHandle_t& cublas_handle,
            cudnnTensorDescriptor_t data_tensor_desc_p,
            uint seed = 0);

    void add_layer(Layer* layer);
    cudnnTensorDescriptor_t last_layer_outp_desc();

    void set_metric(MetricLayer* metric);

    void fit(TrainData& data, size_t epoches, float lr, float momentum);
    std::vector<int> predict_labels(TestData& data);


    std::vector<Layer*> layers;
    MetricLayer* metric;

private:

    cudnnHandle_t& cudnn_handle;
    cublasHandle_t& cublas_handle;
    cudnnTensorDescriptor_t data_tensor_desc;

};


#endif //CUDNN_PROJ_CONVNET_CUH
