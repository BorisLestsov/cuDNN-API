#include "Data.cuh"

Data::Data(cudnnHandle_t& cudnn_handle_p, const char* in_img_fname, const char* in_nms_fname, size_t batch_size):
        cudnn_handle(cudnn_handle_p),
        batch_size(batch_size),
        loaded(0),
        n_read(0)
{
    _in_f_data.open(in_img_fname, std::ifstream::binary | std::ifstream::in);
    if (!_in_f_data.good())
        throw std::runtime_error("Could not open file with data");
    _in_f_ids.open(in_nms_fname, std::ifstream::binary | std::ifstream::in);
    if (!_in_f_ids.good())
        throw std::runtime_error("Could not open file with ids");

    _in_f_data.read((char*) &n_examples, sizeof(int32_t));
    _in_f_data.read((char*) &ex_H, sizeof(int32_t));
    _in_f_data.read((char*) &ex_W, sizeof(int32_t));
    _in_f_data.read((char*) &ex_C, sizeof(int32_t));

    _ex_size_bytes = ex_H * ex_W * ex_C * sizeof(float);
    _batch_size_bytes = _ex_size_bytes * batch_size;

    img_data = (float*) malloc(_batch_size_bytes);
    ids_data = (ids_t*) malloc(batch_size * sizeof(int32_t));


    checkCudaErrors( cudaMalloc(&d_img_data, _batch_size_bytes) );
    checkCudnnErrors( cudnnCreateTensorDescriptor(&img_data_tensor_desc) );
    checkCudnnErrors( cudnnSetTensor4dDescriptor(img_data_tensor_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch_size, ex_C,
                                           ex_H, ex_W) );
}

Data::~Data(){
    _in_f_data.close();
    _in_f_ids.close();
    free(img_data);
    free(ids_data);

    checkCudaErrors( cudaFree(d_img_data) );
    cudnnDestroyTensorDescriptor(img_data_tensor_desc);
}

uint Data::ex_left(){
    return n_examples - n_read;
}

bool Data::is_finished(){
    return n_read == n_examples;
}