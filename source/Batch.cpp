#include "Batch.h"

Batch::Batch(size_t siz, char* data):
        size(siz),
        _data(data),
        _is_labeled(false),
        _labels(Labels())
{}

Batch::Batch(size_t siz, char* data, const Labels& lb):
        size(siz),
        _data(data),
        _is_labeled(true),
        _labels(lb)
{
}

bool Batch::is_labeled(){
    return _is_labeled;
}


Labels& Batch::get_labels(){
    return _labels;
}


void Batch::set_labels(const Labels& lb){
    _labels = lb;
}