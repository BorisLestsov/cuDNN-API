#ifndef CUDNN_PROJ_CONVOLUTIONLAYER_H
#define CUDNN_PROJ_CONVOLUTIONLAYER_H

#include "Layer.cuh"
#include "cstdlib"

#include <random>


class ConvolutionLayer: public Layer {

public:
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;


    cudnnTensorDescriptor_t input_tensor_desc;
    cudnnTensorDescriptor_t output_tensor_desc;
    cudnnTensorDescriptor_t convbias_tensor_desc;

    cudnnConvolutionFwdAlgo_t algo;

    int in_N, in_C, in_H, in_W;
    int depth, kernel_size, filter_stride, zero_padding;
    int out_N, out_C, out_H, out_W;

    size_t workspace_size_bytes;
    size_t weights_length;
    size_t output_length;

    float* h_weights, *h_bias;
    float* d_weights, *d_bias;
    float* d_output;

    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p);
    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
                     cudnnTensorDescriptor_t input_tensor_desc_p,
                     size_t depth, size_t ker_size, size_t stride, size_t zp = 0);

    ~ConvolutionLayer();


    void init_weights_random(std::mt19937& gen);
    void load_weights_from_file(const char* fname);

    void propagate_forward(float* d_x);
    void propagate_backward(float* d_dy, float* d_x);
    void update_weights(float lr);

private:
    cudnnHandle_t& cudnn_handle;
    void* _workspace_forward;

    float _randrange;
};


#endif //CUDNN_PROJ_CONVOLUTIONLAYER_H
