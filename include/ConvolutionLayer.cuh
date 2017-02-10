#ifndef CUDNN_PROJ_CONVOLUTIONLAYER_H
#define CUDNN_PROJ_CONVOLUTIONLAYER_H

#include "Layer.cuh"
#include "cstdlib"

class ConvolutionLayer: public Layer {

public:
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;

    // TODO: Replace these with costructor parameters

    size_t in_N, in_H, in_W, in_C;
    size_t depth, kernel_size, filter_stride, zero_padding;
    size_t out_N, out_H, out_W, out_C;  // FORWARD!!!



    float* h_weights, *h_bias;
    float* d_weights, *d_bias;

    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p);
    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
                     cudnnTensorDescriptor_t data_tensor_desc_p,
                     size_t depth, size_t ker_size, size_t stride, size_t zp = 0);
    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
                     size_t in_C, size_t out_C, size_t kernel_size,
                     size_t in_W, size_t in_H,
                     size_t out_W, size_t out_H);
    ~ConvolutionLayer();


    void init_weights_random(/* rand function?*/);
    void load_weights_from_file(const char* fname);

    // TODO: ???
    void propagate_forward(float* d_x);

private:
    cudnnHandle_t& cudnn_handle;

};


#endif //CUDNN_PROJ_CONVOLUTIONLAYER_H
