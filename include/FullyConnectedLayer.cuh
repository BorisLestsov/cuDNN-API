#ifndef CUDNN_PROJ_FULLYCONNECTEDLAYER_H
#define CUDNN_PROJ_FULLYCONNECTEDLAYER_H

#include "Layer.cuh"
#include "cstdlib"

class FullyConnectedLayer: public Layer {
public:
    cudnnTensorDescriptor_t input_tensor_desc;
    cudnnTensorDescriptor_t output_tensor_desc;

    cudnnDataType_t inp_datatype;

    int in_N, in_C, in_H, in_W;
    size_t n_inp, n_outp;
    int out_N, out_C, out_H, out_W;

    float* h_weights, *h_bias;
    float* d_weights, *d_bias;

    float* d_output;

    float* d_grad_w, *d_grad_b;
    float* d_dx;


    FullyConnectedLayer(cublasHandle_t& cudnn_handle_p,
                        cudnnTensorDescriptor_t input_tensor_desc_p, size_t n_outputs_p);
    ~FullyConnectedLayer();


    void init_weights_random(std::mt19937& gen);
    void load_weights_from_file(const char* fname);

    void propagate_forward(float* d_x);
    void propagate_backward(float* d_dy, float* d_x);
    void update_weights(float lr);

private:
    cublasHandle_t& cublas_handle;

    float* h_ones;
    float* d_ones;

    float _randrange;

};


#endif //CUDNN_PROJ_FULLYCONNECTEDLAYER_H
