#ifndef CUDNN_PROJ_DATA_CUH
#define CUDNN_PROJ_DATA_CUH

#include "types.h"
#include "helper_functions.cuh"

#include <iostream>
#include <fstream>
#include <exception>
#include <cstddef>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cudnn.h>

using std::ifstream;

class Data {
public:
    cudnnHandle_t& cudnn_handle;

    int32_t n_examples;
    int32_t ex_H;
    int32_t ex_W;
    int32_t ex_C;

    const size_t batch_size;
    size_t loaded;

    cudnnTensorDescriptor_t img_data_tensor_desc;

    float* img_data;
    ids_t* ids_data;

    // GPU data
    float* d_img_data;


    Data(cudnnHandle_t& cudnn_handle_p,
         const char* in_img_fname,
         const char* in_nms_fname,
         size_t batch_size);
    ~Data();

    bool is_finished();
    uint ex_left();

    virtual void load_next_batch() = 0;
    virtual void copy_batch_to_GPU() = 0;


protected:

    ifstream _in_f_data;
    ifstream _in_f_ids;

    int32_t n_read;

    size_t _ex_size_bytes;
    size_t _batch_size_bytes;
};


#endif //CUDNN_PROJ_DATA_CUH
