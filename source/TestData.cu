#include "TestData.cuh"

TestData::TestData(cudnnHandle_t& cudnn_handle_p,
                   const char* data_f,
                   const char* ids_f,
                   int n_labels_p,
                   size_t batch_size) :
        Data(cudnn_handle_p, data_f, ids_f, batch_size),
        n_labels(n_labels_p)
{
    h_lbl_data = (float*) malloc(batch_size * n_labels * sizeof(float));
    predicted_labels = (int*) malloc(n_examples * sizeof(int));
}

TestData::~TestData()
{
    free(h_lbl_data);
    free(predicted_labels);
}


void TestData::load_next_batch() {
    int32_t ex_to_read = ((n_examples - n_read) > batch_size )? batch_size
                                                              :(n_examples - n_read);
    int32_t bytes_to_read = ex_to_read * _ex_size_bytes;

    _in_f_data.read( (char*) img_data, bytes_to_read);
    uint bytes_read;
    bytes_read = _in_f_data.gcount();

    if (bytes_read != bytes_to_read)
        throw std::runtime_error("Image data read error");

    _in_f_ids.read( (char*) ids_data, ex_to_read * sizeof(int32_t) );
    bytes_read = _in_f_ids.gcount();
    if (bytes_read != ex_to_read * sizeof(int32_t))
        throw std::runtime_error("Ids data read error");

    n_read += ex_to_read;
    loaded = ex_to_read;

}

void TestData::copy_batch_to_GPU(){
    checkCudaErrors( cudaMemcpyAsync(d_img_data, img_data,
                                     loaded * _ex_size_bytes, cudaMemcpyHostToDevice) );
}

void TestData::reset(){
    _in_f_data.seekg(4*sizeof(int32_t), _in_f_data.beg);
    _in_f_ids.seekg(0, _in_f_ids.beg);
    n_read = 0;
    loaded = 0;
}


std::vector<int> TestData::predict_batch_classes(float *d_sm_output){

    static std::vector<int> tmp(batch_size);

    checkCudaErrors( cudaMemcpyAsync(h_lbl_data, d_sm_output,
                                     sizeof(float) * n_labels * batch_size, cudaMemcpyDeviceToHost) );

    for (uint ex = 0; ex < batch_size; ++ex){
        float max_label_val = 0.0;
        int max_label_ind = 0;

        for (uint label = 0; label < n_labels; ++label){
            if(h_lbl_data[ex*batch_size + label] > max_label_val){
                max_label_val = h_lbl_data[ex*batch_size + label];
                max_label_ind = label;
            }
        }
        tmp[ex] = max_label_ind;

        //std::cout << "Example " << ids_data[ex] << ": " << max_label_ind << "    " << max_label_val << std::endl;
    }

    return tmp;
}