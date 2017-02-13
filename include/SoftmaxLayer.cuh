#ifndef CUDNN_PROJ_SOFTMAXLAYER_H
#define CUDNN_PROJ_SOFTMAXLAYER_H


#include "Layer.cuh"


class SoftmaxLayer: public Layer {
public:
    cudnnTensorDescriptor_t input_tensor_desc;
    cudnnTensorDescriptor_t output_tensor_desc;

    cudnnDataType_t inp_datatype;

    // TODO: Replace these with costructor parameters

    const int output_tensor_dims = 4;

    float* d_output;

    int in_N, in_C, in_H, in_W;
    int out_N, out_C, out_H, out_W;  // FORWARD!!!

    SoftmaxLayer(cudnnHandle_t& cudnn_handle_p);
    SoftmaxLayer(cudnnHandle_t& cudnn_handle_p,
                 cudnnTensorDescriptor_t input_tensor_desc_p);

    ~SoftmaxLayer();

    // TODO: ???
    void propagate_forward(float* d_x);

private:
    cudnnHandle_t& cudnn_handle;


};


#endif //CUDNN_PROJ_SOFTMAXLAYER_H
