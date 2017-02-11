#ifndef CUDNN_PROJ_CONVOLUTIONLAYER_H
#define CUDNN_PROJ_CONVOLUTIONLAYER_H

#include "Layer.cuh"
#include "cstdlib"

class ConvolutionLayer: public Layer {

public:
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnFilterDescriptor_t filter_desc;

    cudnnTensorDescriptor_t input_tensor_desc;
    cudnnTensorDescriptor_t output_tensor_desc;

    cudnnConvolutionFwdAlgo_t algo;

    // TODO: Replace these with costructor parameters

    int in_N, in_C, in_H, in_W;
    int depth, kernel_size, filter_stride, zero_padding;
    int out_N, out_C, out_H, out_W;  // FORWARD!!!

    size_t workspace_size_bytes;

    float* h_weights, *h_bias;
    float* d_weights, *d_bias;
    float* d_output;

    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p);
    ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
                     cudnnTensorDescriptor_t input_tensor_desc_p,
                     size_t depth, size_t ker_size, size_t stride, size_t zp = 0);

    ~ConvolutionLayer();


    void init_weights_random(/* rand function?*/);
    void load_weights_from_file(const char* fname);

    // TODO: ???
    void propagate_forward(float* d_x);

private:
    cudnnHandle_t& cudnn_handle;
    void* _workspace;


};


#endif //CUDNN_PROJ_CONVOLUTIONLAYER_H
