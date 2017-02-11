#include "ConvolutionLayer.cuh"

ConvolutionLayer::ConvolutionLayer(cudnnHandle_t& cudnn_handle_p,
                cudnnTensorDescriptor_t input_tensor_desc_p,
                size_t depth_p,
                size_t ker_size,
                size_t stride,
                size_t zp):
        cudnn_handle(cudnn_handle_p),
        input_tensor_desc(input_tensor_desc_p),
        depth(depth_p),
        kernel_size(ker_size),
        filter_stride(stride),
        zero_padding(zp),
        in_C(3),
        out_C(depth_p)
{
    checkCudnnErrors( cudnnCreateFilterDescriptor(&filter_desc) );
    checkCudnnErrors( cudnnCreateConvolutionDescriptor(&conv_desc) );

    cudnnDataType_t inp_datatype;
    int inp_strid;
    checkCudnnErrors( cudnnGetTensor4dDescriptor(input_tensor_desc,
                                                 &inp_datatype,
                                                 &in_N, &in_C, &in_H, &in_W,
                                                 &inp_strid, &inp_strid, &inp_strid, &inp_strid) );

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
                                                            input_tensor_desc,
                                                            filter_desc,
                                                            tensor_dims,
                                                            output_tensor_dims) );
    out_N = output_tensor_dims[0];
    out_C = output_tensor_dims[1];
    out_H = output_tensor_dims[2];
    out_W = output_tensor_dims[3];

    std::cout << "Conv output forward dims:" << std::endl;
    for (uint i = 0; i < tensor_dims; ++i){
        std::cout << output_tensor_dims[i] << "  ";
    }
    std::cout << std::endl;

    checkCudnnErrors( cudnnCreateTensorDescriptor(&output_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(output_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 out_N, out_C,
                                                 out_H, out_W) );


    checkCudnnErrors( cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                          input_tensor_desc,
                                                          filter_desc,
                                                          conv_desc,
                                                          output_tensor_desc,
                                                          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                          0,
                                                          &algo) );

    checkCudnnErrors( cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                              input_tensor_desc,
                                                              filter_desc,
                                                              conv_desc,
                                                              output_tensor_desc,
                                                              algo,
                                                              &workspace_size_bytes) );
    std::cout << "Workspace size: " << workspace_size_bytes << std::endl;

    checkCudaErrors( cudaMalloc(&_workspace, workspace_size_bytes) );
    checkCudaErrors( cudaMalloc(&d_weights, sizeof(float) * in_C * kernel_size * kernel_size * out_C) );
    checkCudaErrors( cudaMalloc(&d_output, sizeof(float) * in_N * out_C * out_H * out_W) );

}



ConvolutionLayer::~ConvolutionLayer() {
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyTensorDescriptor(output_tensor_desc);

    checkCudaErrors( cudaFree(d_weights) );
    checkCudaErrors( cudaFree(d_output) );
    checkCudaErrors( cudaFree(_workspace) );
}


void ConvolutionLayer::propagate_forward(float* d_x){
    float alpha = 1.0f;
    float beta = 0.0f;

    checkCudnnErrors( cudnnConvolutionForward(cudnn_handle, &alpha, input_tensor_desc,
                                              d_x, filter_desc, d_weights, conv_desc,
                                              algo, _workspace, workspace_size_bytes, &beta,
                                              output_tensor_desc, d_output) );
}

