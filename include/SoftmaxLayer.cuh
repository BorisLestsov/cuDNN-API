#ifndef CUDNN_PROJ_SOFTMAXLAYER_H
#define CUDNN_PROJ_SOFTMAXLAYER_H


#include "Layer.cuh"


class SoftmaxLayer: public Layer {
public:
    const int output_tensor_dims = 4;

    int n_labels;

    SoftmaxLayer(cudnnHandle_t& cudnn_handle_p,
                 cudnnTensorDescriptor_t input_tensor_desc_p);

    ~SoftmaxLayer();

    void propagate_forward(float* d_x) override;
    void propagate_backward(float* d_targ, float* d_dx, float momentum = 0.0) override;
    void update_weights(float lr) override {};

private:
};


#endif //CUDNN_PROJ_SOFTMAXLAYER_H
