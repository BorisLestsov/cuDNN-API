#include "Data.h"

Data::Data(const char* in_img_fname, const char* in_nms_fname, size_t batch_size):
        batch_size(batch_size),
        n_read(0)
{
    _in_f_data.open(in_img_fname, std::ifstream::binary);
    if (!_in_f_data.good())
        throw std::runtime_error("Could not open file with data");
    _in_f_ids.open(in_nms_fname, std::ifstream::binary);
    if (!_in_f_ids.good())
        throw std::runtime_error("Could not open file with ids");

    _in_f_data.read((char*) &n_examples, sizeof(int32_t));
    _in_f_data.read((char*) &ex_H, sizeof(int32_t));
    _in_f_data.read((char*) &ex_W, sizeof(int32_t));
    _in_f_data.read((char*) &ex_C, sizeof(int32_t));

    _ex_size_bytes = ex_H * ex_W * ex_C;
    _batch_size_bytes = _ex_size_bytes * batch_size;

    Batch::_ex_size_bytes = _ex_size_bytes;
}

Data::~Data(){
    _in_f_data.close();
}

uint Data::ex_left(){
    return n_examples - n_read;
}

bool Data::is_finished(){
    return n_read == n_examples;
}