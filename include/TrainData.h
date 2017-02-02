#ifndef CUDNN_PROJ_TRAINDATA_H
#define CUDNN_PROJ_TRAINDATA_H

#include "Data.h"

class TrainData: public Data {
public:
    TrainData(
            const char* data_f,
            const char* labels_f,
            size_t batch_size,
            size_t ex_H,
            size_t ex_W,
            size_t ex_C);

    Batch get_next_batch() override;

private:

    ifstream _in_f_labels;

};


#endif //CUDNN_PROJ_TRAINDATA_H
