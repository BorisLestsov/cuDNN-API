#ifndef CUDNN_PROJ_BATCH_H
#define CUDNN_PROJ_BATCH_H

#include "types.h"

#include <iostream>
#include <fstream>
#include <exception>
#include <cstddef>
#include <cstdlib>


class Batch {
public:
    const size_t size;


    Batch(size_t siz, char* data);
    Batch(size_t siz, char* data, const Labels& lb);

    bool is_labeled();
    Labels& get_labels();
    void set_labels(const Labels& lb);

private:
    char* _data;

    bool _is_labeled;
    Labels _labels;
};


#endif //CUDNN_PROJ_BATCH_H
