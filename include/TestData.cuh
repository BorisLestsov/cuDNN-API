#ifndef CUDNN_PROJ_TESTDATA_CUH
#define CUDNN_PROJ_TESTDATA_CUH

#include "Data.cuh"

class TestData: public Data {
public:

    int n_labels;

    float* h_lbl_data;
    float* d_lbl_data;
    int* predicted_labels;

    TestData(cudnnHandle_t& cudnn_handle_p,
              const char* data_f,
              const char* ids_f,
              int n_labels_p,
              size_t batch_size);
    ~TestData();


    void load_next_batch() override;
    void copy_batch_to_GPU() override;
    void reset();

    void predict_batch_classes(float* d_sm_output);

    void predict_batch_classes_proba(float* d_sm_output);

private:
    void update_labels();

    //TODO: ofstream _out_f_labels; - ???

};


#endif //CUDNN_PROJ_TESTDATA_CUH
