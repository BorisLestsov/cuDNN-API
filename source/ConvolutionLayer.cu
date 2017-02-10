#include "ConvolutionLayer.cuh"

ConvolutionLayer::ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
                cudnnTensorDescriptor_t data_tensor_desc_p,
                size_t depth_p,
                size_t ker_size,
                size_t stride,
                size_t zp):
        cudnn_handle(cudnn_handle_p),
        depth(depth_p),
        kernel_size(ker_size),
        filter_stride(stride),
        zero_padding(zp),
        in_C(3),
        out_C(3)
{
    checkCudnnErrors( cudnnCreateFilterDescriptor(&filter_desc) );
    checkCudnnErrors( cudnnCreateConvolutionDescriptor(&conv_desc) );

    const size_t conv_dims = 2;
    const int pad[conv_dims] = {0, 0};
    const int strides[conv_dims] = {filter_stride, filter_stride};
    const int upscale[conv_dims] = {1, 1};

    checkCudnnErrors( cudnnSetFilter4dDescriptor(filter_desc,
                                                 CUDNN_DATA_FLOAT,
                                                 CUDNN_TENSOR_NCHW,
                                                 out_C,
                                                 in_C,
                                                 kernel_size,
                                                 kernel_size) );

    checkCudnnErrors( cudnnSetConvolutionNdDescriptor(conv_desc,
                                                      conv_dims,
                                                      pad,
                                                      strides,
                                                      upscale,
                                                      CUDNN_CROSS_CORRELATION,
                                                      CUDNN_DATA_FLOAT) );

    int tensor_dims = 4;
    int output_tensor_dims[tensor_dims];
    checkCudnnErrors( cudnnGetConvolutionNdForwardOutputDim(conv_desc,
                                                           data_tensor_desc_p,
                                                           filter_desc,
                                                           tensor_dims,
                                                           output_tensor_dims) );

    for (uint i = 0; i < tensor_dims; ++i){
        std::cout << output_tensor_dims[i] << "    ";
    }
    std::cout << std::endl;
}

/*
ConvolutionLayer::ConvolutionLayer(cudnnHandle_t& cudnn_handle_p):
        cudnn_handle(cudnn_handle_p),

{
    checkCudnnErrors( cudnnCreateTensorDescriptor(&filter_desc) );
    checkCudnnErrors( cudnnCreateTensorDescriptor(&conv_desc) );

    checkCudnnErrors(cudnnSetFilter4dDescriptor(filterDesc,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          out_C,
                                          in_C,
                                          kernel_size,
                                          kernel_size));

    checkCudnnErrors( cudnnSetConvolutionNdDescriptor(convDesc,
                                                      convDims,
                                                      pad,
                                                      filterStride,
                                                      upscale,
                                                      CUDNN_CROSS_CORELLATION) );
    checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                     srcTensorDesc,
                                                     filterDesc,
                                                     &out_N, &out_C, &out_H, &out_W));
}


ConvolutionLayer::ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
size_t in_C, size_t out_C, size_t kernel_size,
        size_t in_W, size_t in_H,
        size_t out_W, size_t out_H);
*/


ConvolutionLayer::~ConvolutionLayer() {
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
}
