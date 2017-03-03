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
        out_C(depth_p),
        _randrange(0.01)
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

    checkCudnnErrors( cudnnCreateTensorDescriptor(&convbias_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(convbias_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, out_C,
                                                 1, 1) );


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
    //std::cout << "Workspace size: " << workspace_size_bytes << std::endl;

    checkCudaErrors( cudaMalloc(&_workspace_forward, workspace_size_bytes) );

    weights_length = in_N * kernel_size * kernel_size * out_C;
    output_length = out_N * out_C * out_H * out_W;

    h_weights = (float*) malloc(sizeof(float) * weights_length);
    h_bias = (float*) malloc(sizeof(float) * out_C);

    checkCudaErrors( cudaMalloc(&d_weights, sizeof(float) * weights_length) );
    checkCudaErrors( cudaMalloc(&d_bias, sizeof(float) * out_C) );
    checkCudaErrors( cudaMalloc(&d_output, sizeof(float) * output_length) );

}



ConvolutionLayer::~ConvolutionLayer() {
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyTensorDescriptor(output_tensor_desc);
    cudnnDestroyTensorDescriptor(convbias_tensor_desc);
    //TODO: check tensor desc copy

    free(h_weights);
    free(h_bias);

    checkCudaErrors( cudaFree(d_weights) );
    checkCudaErrors( cudaFree(d_bias) );
    checkCudaErrors( cudaFree(d_output) );
    checkCudaErrors( cudaFree(_workspace_forward) );
}


void ConvolutionLayer::propagate_forward(float* d_x){
    float alpha = 1.0f;
    float beta = 0.0f;


    checkCudnnErrors( cudnnConvolutionForward(cudnn_handle,
                                              &alpha,
                                              input_tensor_desc, d_x,
                                              filter_desc, d_weights,
                                              conv_desc, algo,
                                              _workspace_forward, workspace_size_bytes,
                                              &beta,
                                              output_tensor_desc, d_output) );

    checkCudnnErrors( cudnnAddTensor(cudnn_handle,
                                     &alpha,
                                     convbias_tensor_desc, d_bias,
                                     &alpha,
                                     output_tensor_desc, d_output) );

    /*
    cudnnTensorDescriptor_t in1;
    checkCudnnErrors( cudnnCreateTensorDescriptor(&in1) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(in1,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, in_C, in_H, in_W) );


    This may work for classification of 1 example

    cudnnTensorDescriptor_t out1;
    checkCudnnErrors( cudnnCreateTensorDescriptor(&out1) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(out1,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, 96, 55, 55) );

    checkCudnnErrors( cudnnConvolutionForward(cudnn_handle, &alpha, in1,
                                              d_x, filter_desc, d_weights, conv_desc,
                                              algo, _workspace_forward, workspace_size_bytes, &beta,
                                              out1, d_output) );
                                              */
}

void ConvolutionLayer::propagate_backward(float* d_dy, float* d_x) {
    float alpha = 1.0f;
    float beta = 0.0f;


}


void ConvolutionLayer::update_weights(float lr){

}


void ConvolutionLayer::init_weights_random(std::mt19937& gen){
    std::uniform_real_distribution<> get_rand(-_randrange, _randrange);

    for (uint i = 0; i < weights_length; ++i)
        h_weights[i] = static_cast<float>(get_rand(gen));

    for (uint i = 0; i < out_C; ++i)
        h_bias[i] = 1.0f;

    checkCudaErrors( cudaMemcpy(d_weights, h_weights,
                                sizeof(float) * weights_length, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_bias, h_bias,
                                sizeof(float) * out_C, cudaMemcpyHostToDevice) );
}
