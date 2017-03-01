#ifndef CUDNN_PROJ_TRAINDATA_CUH
#define CUDNN_PROJ_TRAINDATA_CUH

#include "Data.cuh"

class TrainData: public Data {
public:

    int n_labels;

    float* lbl_data;
    float* d_lbl_data;

    TrainData(cudnnHandle_t& cudnn_handle_p,
              const char* data_f,
              const char* ids_f,
              const char* labels_f,
              size_t batch_size);
    ~TrainData();


    void load_next_batch() override;
    void copy_batch_to_GPU() override;
    void reset();

private:

    ifstream _in_f_labels;

};


#endif //CUDNN_PROJ_TRAINDATA_CUH
