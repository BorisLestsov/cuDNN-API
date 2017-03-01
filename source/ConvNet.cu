#include "ConvNet.cuh"


ConvNet::ConvNet(cudnnHandle_t& cudnn_handle_p,
                 cublasHandle_t& cublas_handle_p,
                 cudnnTensorDescriptor_t data_tensor_desc_p,
                 uint seed):

        cudnn_handle(cudnn_handle_p),
        cublas_handle(cublas_handle_p),
        data_tensor_desc(data_tensor_desc_p),


        fc1(cublas_handle_p, data_tensor_desc_p, 90),
        sm(cudnn_handle_p, fc1.output_tensor_desc),
        mse(cudnn_handle_p, fc1.output_tensor_desc),

        gen(seed == 0 ? rd() : seed)
{
    fc1.init_weights_random(gen);
}


void ConvNet::fit(TrainData& train, uint epoches, float lr){

    /*float* h_dy = (float*) calloc(sm.out_N * sm.out_C * sm.out_H * sm.out_W, sizeof(float));
    float* lbls = (float*) calloc(sm.out_N * sm.out_C * sm.out_H * sm.out_W, sizeof(float));
    lbls[0] = 1.0f;
    lbls[sm.out_C * sm.out_H * sm.out_W] = 1.0f;
*/

//    float x[4] = {2.0, 0.0, 5.5, 1.0};
//    float t[4] = {0.0, 1.0, 1.5, 0.0};
//    float* d_x;
//    float* d_t;
//
//    cudaMalloc(&d_x, 4*sizeof(float));
//    cudaMalloc(&d_t, 4*sizeof(float));
//
//    checkCudaErrors(cudaMemcpy(d_x, x,
//                               4 * sizeof(float), cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpy(d_t, t,
//                               4 * sizeof(float), cudaMemcpyHostToDevice));

    for (uint ep = 0; ep < epoches; ++ep) {
        std::cout << "Epoch: " << ep << std::endl;
        while (!train.is_finished()) {
            std::cout << "Propagating next batch: " << train.get_n_read() << std::endl;

            train.load_next_batch();
            train.copy_batch_to_GPU();


            fc1.propagate_forward(train.d_img_data);
            mse.propagate_forward(train.d_lbl_data, fc1.d_output);

            mse.propagate_backward(train.d_lbl_data, fc1.d_output);
            fc1.propagate_backward(mse.d_dx, train.d_img_data);

            //fc1.propagate_forward(train.d_img_data);
            //sm.propagate_forward(fc1.d_output);

            //sm.propagate_backward(train.d_lbl_data, fc1.d_output);
            //fc1.propagate_backward(sm.d_dx, train.d_img_data);
            /*checkCudaErrors(cudaMemcpy(d_y, fc1.d_output,
                                       in_N * in_C * in_H * in_W * sizeof(float), cudaMemcpyDeviceToDevice));
    */
        }
    }

}


char* ConvNet::predict(TestData&){
    return nullptr;
}
