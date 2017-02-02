#include "TrainData.h"

TrainData::TrainData(
        const char* data_f,
        const char* labels_f,
        size_t batch_size,
        size_t ex_H,
        size_t ex_W,
        size_t ex_C)
:
        Data(data_f, batch_size, ex_H, ex_W, ex_C)
{
    _in_f_labels.open(labels_f, std::ifstream::binary);
    if (!_in_f_labels.good())
        throw std::runtime_error("Could not open file with labels");

    Data::_labels.resize(batch_size);
}


Batch TrainData::get_next_batch() {
    _in_f_data.read(_data, _batch_size_bytes);
    uint _bytes_read;
    _bytes_read = _in_f_data.gcount();
    if (_bytes_read == 0)
        return Batch(0, nullptr);

    if (_bytes_read % _ex_size_bytes != 0)
        throw std::runtime_error("Image data read error");

    uint _examples_read = _bytes_read / _ex_size_bytes;

    _in_f_labels.read(_labels.data(), _examples_read);
    _bytes_read = _in_f_labels.gcount();
    if (_bytes_read != _examples_read)
        throw std::runtime_error("Labels data read error");

    Batch res(_examples_read, _data, _labels);

    return res;
}