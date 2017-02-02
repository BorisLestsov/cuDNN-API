#ifndef CUDNN_PROJ_DATA_H
#define CUDNN_PROJ_DATA_H

#include "Batch.h"
#include "types.h"

#include <iostream>
#include <fstream>
#include <exception>
#include <cstddef>
#include <cstdlib>

using std::ifstream;

class Data {
public:
    const size_t batch_size;
    const size_t ex_H;
    const size_t ex_W;
    const size_t ex_C;


    Data(const char* in_fname, size_t batch_size, size_t ex_H, size_t ex_W, size_t ex_C);
    ~Data();


    virtual Batch get_next_batch() = 0;


protected:
    char* _data;

    size_t _ex_size_bytes;
    size_t _batch_size_bytes;

    Labels _labels;
    ifstream _in_f_data;

};


#endif //CUDNN_PROJ_DATA_H
