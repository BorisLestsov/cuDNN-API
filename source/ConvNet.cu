#include "ConvNet.cuh"


ConvNet::ConvNet(cudnnHandle_t& cudnn_handle_p,
                 cublasHandle_t& cublas_handle_p,
                 cudnnTensorDescriptor_t data_tensor_desc_p):

        cudnn_handle(cudnn_handle_p),
        cublas_handle(cublas_handle_p),
        fc1(cublas_handle_p, 2, 3),
        conv1(cudnn_handle_p, data_tensor_desc_p, 96, 11, 4),
        data_tensor_desc(data_tensor_desc_p)
{
    fc1.init_weights_random();
}


void ConvNet::fit(TrainData& train){
    while (!train.is_finished()){
        train.load_next_batch();
        for (uint i = 0; i < train.loaded; ++i){
            std::cout << train.ids_data[i] << "   " << train.lbl_data[i] << std::endl;
        }
        std::cout << std::endl;
    }

    conv1.propagate_forward(train.d_img_data);

}


char* ConvNet::predict(TestData&){
    return nullptr;
}
