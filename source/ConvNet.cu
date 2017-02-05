#include "ConvNet.cuh"


ConvNet::ConvNet(cudnnHandle_t& cudnn_handle_p, cublasHandle_t& cublas_handle_p):
        cudnn_handle(cudnn_handle_p),
        cublas_handle(cublas_handle_p)
{}


void ConvNet::fit(TrainData& train){
    while (!train.is_finished()){
        train.load_next_batch();
        for (uint i = 0; i < train.loaded; ++i){
            std::cout << train.ids_data[i] << "   " << train.lbl_data[i] << std::endl;
        }
        std::cout << std::endl;
    }
}


char* ConvNet::predict(TestData&){
    return nullptr;
}