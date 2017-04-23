#ifndef CUDNN_PROJ_CONVOLUTIONLAYER_H
#define CUDNN_PROJ_CONVOLUTIONLAYER_H

#include "Layer.cuh"
#include "cstdlib"

#include <random>
#include <fstream>


class ConvolutionLayer: public Layer {

public:
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;

    cudnnTensorDescriptor_t bias_tensor_desc;

    cudnnConvolutionFwdAlgo_t forward_algo;
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    cudnnConvolutionBwdDataAlgo_t data_algo;

    int depth, kernel_size, filter_stride, zero_padding;

    size_t workspace_size_bytes;
    size_t weights_length;
    size_t output_length;
    size_t bias_length;

    float* h_weights, *h_bias;
    float* d_weights, *d_bias;

    float* d_dbias, *d_dweights;

    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
                     cublasHandle_t& cublas_handle_p,
                     cudnnTensorDescriptor_t input_tensor_desc_p,
                     size_t depth, size_t ker_size, size_t stride, size_t zp = 0);

    ~ConvolutionLayer();


    void init_weights_random(std::mt19937& gen);
    void load_weights_from_file(const char* fname);
    void save_kernels(const char* fname);

    void propagate_forward(float* d_x) override;
    void propagate_backward(float* d_dy, float* d_x, float momentum = 0.9) override;
    void update_weights(float lr) override;

private:

    void* d_workspace;

    float _randrange;
};


#endif //CUDNN_PROJ_CONVOLUTIONLAYER_H
