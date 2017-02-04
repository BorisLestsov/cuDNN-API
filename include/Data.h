#ifndef CUDNN_PROJ_DATA_H
#define CUDNN_PROJ_DATA_H

#include "types.h"

#include <iostream>
#include <fstream>
#include <exception>
#include <cstddef>
#include <cstdlib>

using std::ifstream;

class Data {
public:
    ids_t n_examples;
    ids_t ex_H;
    ids_t ex_W;
    ids_t ex_C;

    const size_t batch_size;
    size_t loaded;

    float* img_data;
    ids_t* ids_data;

    Data(const char* in_img_fname, const char* in_nms_fname, size_t batch_size);
    ~Data();

    bool is_finished();
    uint ex_left();

    virtual void load_next_batch() = 0;


protected:

    ifstream _in_f_data;
    ifstream _in_f_ids;

    ids_t n_read;

    size_t _ex_size_bytes;
    size_t _batch_size_bytes;
};


#endif //CUDNN_PROJ_DATA_H
