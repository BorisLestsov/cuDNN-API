#ifndef CUDNN_PROJ_LAYER_H
#define CUDNN_PROJ_LAYER_H

#include "helper_functions.cuh"

#include <random>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>


enum class Layer_t {Convolution, Activation, Pooling, FullyConnected, Softmax, NLL, MSE};

class Layer {
public:

    Layer_t type;

    Layer(Layer_t type,
          cudnnTensorDescriptor_t input_tensor_desc,
          cudnnHandle_t cudnn_handle,
          cublasHandle_t cublas_handle);

    cudnnDataType_t inp_datatype;

    cudnnTensorDescriptor_t input_tensor_desc;
    cudnnTensorDescriptor_t output_tensor_desc;

    int in_N, in_C, in_H, in_W;
    int out_N, out_C, out_H, out_W;

    virtual void propagate_forward(float* d_x) = 0;
    virtual void propagate_backward(float* d_dy, float* d_x, float momentum = 0.9) = 0;
    virtual void update_weights(float lr) = 0;
    //virtual void init_weights_random(std::mt19937& gen) = 0;

    float* d_output;
    float* d_dx;


protected:
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
};


class MetricLayer: public Layer {
public:

    float batch_loss;


    MetricLayer(Layer_t type,
                cudnnTensorDescriptor_t input_tensor_desc,
                cudnnHandle_t cudnn_handle,
                cublasHandle_t cublas_handle);

    virtual void compute_loss(float *d_targ, float *d_x) = 0;
};

#endif //CUDNN_PROJ_LAYER_H
