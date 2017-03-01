#include "ConvNet.cuh"


ConvNet::ConvNet(cudnnHandle_t& cudnn_handle_p,
                 cublasHandle_t& cublas_handle_p,
                 cudnnTensorDescriptor_t data_tensor_desc_p,
                 uint seed):

        cudnn_handle(cudnn_handle_p),
        cublas_handle(cublas_handle_p),
        data_tensor_desc(data_tensor_desc_p),


        fc1(cublas_handle_p, data_tensor_desc_p, 50),
        fc2(cublas_handle_p, fc1.output_tensor_desc, 90),
        sm(cudnn_handle_p, fc2.output_tensor_desc),
        mse(cudnn_handle_p, sm.output_tensor_desc),

        gen(seed == 0 ? rd() : seed)
{
    fc1.init_weights_random(gen);
    fc2.init_weights_random(gen);
}


void ConvNet::fit(TrainData& train, int epoches, float lr){

    for (uint ep = 0; ep < epoches; ++ep) {
        std::cout << "Epoch: " << ep << std::endl;
        while (!train.is_finished()) {
            std::cout << "Propagating next batch: " << train.get_n_read() << std::endl;

            train.load_next_batch();
            train.copy_batch_to_GPU();


            fc1.propagate_forward(train.d_img_data);
            fc2.propagate_forward(fc1.d_output);
            sm.propagate_forward(fc2.d_output);
            mse.propagate_forward(train.d_lbl_data, sm.d_output);

            mse.propagate_backward(train.d_lbl_data, sm.d_output);
            sm.propagate_backward(mse.d_dx, fc2.d_output);
            fc2.propagate_backward(sm.d_dx, fc1.d_output);
            fc1.propagate_backward(fc2.d_dx, train.d_img_data);

            fc1.update_weights(lr);
            fc2.update_weights(lr);

        }
        train.reset();
    }

}


char* ConvNet::predict(TestData&){
    return nullptr;
}
