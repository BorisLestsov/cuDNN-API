#ifndef CUDNN_PROJ_FULLYCONNECTEDLAYER_H
#define CUDNN_PROJ_FULLYCONNECTEDLAYER_H

#include "Layer.cuh"
#include "cstdlib"

class FullyConnectedLayer: public Layer {
public:

    size_t n_inp, n_outp;

    float* h_weights, *h_bias;
    float* d_weights, *d_bias;

    float* d_grad_w, *d_grad_b;


    FullyConnectedLayer(cublasHandle_t& cudnn_handle_p,
                        cudnnTensorDescriptor_t input_tensor_desc_p, size_t n_outputs_p);
    ~FullyConnectedLayer();


    void init_weights_random(std::mt19937& gen);
    void load_weights_from_file(const char* fname);

    void propagate_forward(float* d_x) override;
    void propagate_backward(float* d_dy, float* d_x, float momentum = 0.9) override;
    void update_weights(float lr) override;

private:

    size_t weights_length;
    size_t bias_length;

    float* h_ones;
    float* d_ones;

    float _randrange;

};


#endif //CUDNN_PROJ_FULLYCONNECTEDLAYER_H
