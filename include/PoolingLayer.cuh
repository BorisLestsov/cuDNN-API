#ifndef CUDNN_PROJ_POOLINGLAYER_H
#define CUDNN_PROJ_POOLINGLAYER_H

#include "Layer.cuh"
#include "cstdlib"


class PoolingLayer: public Layer {
public:
    cudnnPoolingDescriptor_t pooling_desc;

    cudnnTensorDescriptor_t input_tensor_desc;
    cudnnTensorDescriptor_t output_tensor_desc;

    cudnnPoolingMode_t algo;

    cudnnDataType_t inp_datatype;

    const int output_tensor_dims = 4;

    float* d_output;

    int in_N, in_C, in_H, in_W;
    int size, stride;
    int out_N, out_C, out_H, out_W;  // FORWARD!!!

    PoolingLayer(cudnnHandle_t& cudnn_handle_p);
    PoolingLayer(cudnnHandle_t& cudnn_handle_p,
                cudnnTensorDescriptor_t input_tensor_desc_p,
                        size_t size_p, size_t stride_p);

    ~PoolingLayer();

    void propagate_forward(float* d_x);

private:
    cudnnHandle_t& cudnn_handle;


};


#endif //CUDNN_PROJ_POOLINGLAYER_H
