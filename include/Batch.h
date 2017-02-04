#ifndef CUDNN_PROJ_BATCH_H
#define CUDNN_PROJ_BATCH_H

#include "types.h"

#include <iostream>
#include <fstream>
#include <exception>
#include <cstddef>


class Batch {
public:
    const size_t size;

    Labels labels;
    ImgIds ids;

    Batch(size_t siz, bool with_labels = false);
    Batch(size_t siz, const Labels& lb);

    /*Batch(size_t siz, char* data);
    Batch(size_t siz, char* data, const Labels& lb);*/

    ~Batch();

    bool is_labeled();

    friend class Data;
    friend class TrainData;
    friend class ValidationData;
    friend class TestData;

private:
    static uint _ex_size_bytes;

    char* _data;

    bool _is_labeled;
};


#endif //CUDNN_PROJ_BATCH_H
