#include "Batch.h"

uint Batch::_ex_size_bytes;

Batch::Batch(size_t siz, bool with_labels):
        size(siz),
        _is_labeled(with_labels),
        labels( Labels((with_labels)?siz:0) ),
        ids( ImgIds(siz))
{
    _data = (char*) malloc(_ex_size_bytes * siz);
}

Batch::Batch(size_t siz, const Labels& lb):
        size(siz),
        _is_labeled(true),
        labels(lb),
        ids( ImgIds(siz))
{
    _data = (char*) malloc(_ex_size_bytes * siz);
}


/*
Batch::Batch(size_t siz, char* data):
        size(siz),
        _data(data),
        _is_labeled(false),
        labels(Labels())
{}

Batch::Batch(size_t siz, char* data, const Labels& lb):
        size(siz),
        _data(data),
        _is_labeled(true),
        labels(lb)
{}
*/

Batch::~Batch(){
    free(_data);
}

bool Batch::is_labeled(){
    return _is_labeled;
}
