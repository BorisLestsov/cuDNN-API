#ifndef CUDNN_PROJ_ACTIVATIONLAYER_H
#define CUDNN_PROJ_ACTIVATIONLAYER_H

#include "Layer.cuh"


class ActivationLayer: public Layer {
public:

    cudnnActivationMode_t act_f;
    cudnnActivationDescriptor_t act_desc;

    const int output_tensor_dims = 4;

    ActivationLayer(cudnnHandle_t& cudnn_handle_p,
                    cudnnTensorDescriptor_t input_tensor_desc_p,
                    cudnnActivationMode_t act_f_p);

    ~ActivationLayer();

    void propagate_forward(float* d_x) override;
    void propagate_backward(float* d_dy, float* d_x, float momentum = 0.0) override;
    void update_weights(float lr) override {};
    //void init_weights_random() override {};

private:

};


#endif //CUDNN_PROJ_ACTIVATIONLAYER_H
