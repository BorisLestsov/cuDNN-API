#ifndef CUDNN_PROJ_POOLINGLAYER_H
#define CUDNN_PROJ_POOLINGLAYER_H

#include "Layer.cuh"
#include "cstdlib"


class PoolingLayer: public Layer {
public:
    cudnnPoolingDescriptor_t pooling_desc;

    cudnnPoolingMode_t algo;

    const int output_tensor_dims = 4;

    int size, stride, pad;

    PoolingLayer(cudnnHandle_t& cudnn_handle_p,
                cudnnTensorDescriptor_t input_tensor_desc_p,
                        size_t size_p, size_t stride_p, size_t pad_p);

    ~PoolingLayer();

    void propagate_forward(float* d_x) override;
    void propagate_backward(float* d_dy, float* d_x, float momentum = 0.0) override;
    void update_weights(float lr) override {};

private:

};


#endif //CUDNN_PROJ_POOLINGLAYER_H
