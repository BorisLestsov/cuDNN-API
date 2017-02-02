#include "Data.h"

Data::Data(const char* in_fname, size_t batch_size, size_t ex_H, size_t ex_W, size_t ex_C):
        batch_size(batch_size),
        ex_H(ex_H),
        ex_W(ex_W),
        ex_C(ex_C),
        _ex_size_bytes(ex_H * ex_W * ex_C),
        _batch_size_bytes(_ex_size_bytes * batch_size),
        _labels(Labels())
{
    _in_f_data.open(in_fname, std::ifstream::binary);
    if (!_in_f_data.good())
        throw std::runtime_error("Could not open file with data");

    _data = (char*) malloc(_ex_size_bytes * batch_size);
}

Data::~Data(){
    _in_f_data.close();
    free(_data);
}


// TODO: REMOVE THIS
Batch Data::get_next_batch(){
    _in_f_data.read(_data, _batch_size_bytes);
    uint _bytes_read = _in_f_data.gcount();
    if (_bytes_read % _ex_size_bytes != 0)
        throw std::runtime_error("Data read error");

    return Batch(_bytes_read / _ex_size_bytes, _data);
}