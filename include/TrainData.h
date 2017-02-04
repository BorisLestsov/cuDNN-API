#ifndef CUDNN_PROJ_TRAINDATA_H
#define CUDNN_PROJ_TRAINDATA_H

#include "Data.h"

class TrainData: public Data {
public:

    char* lbl_data;

    TrainData(
            const char* data_f,
            const char* ids_f,
            const char* labels_f,
            size_t batch_size
    );

    ~TrainData();

    void load_next_batch() override;

private:

    ifstream _in_f_labels;

};


#endif //CUDNN_PROJ_TRAINDATA_H
