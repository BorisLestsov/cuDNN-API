#include "ConvNet.cuh"


ConvNet::ConvNet(cudnnHandle_t& cudnn_handle_p,
                 cublasHandle_t& cublas_handle_p,
                 cudnnTensorDescriptor_t data_tensor_desc_p,
                 uint seed):

        cudnn_handle(cudnn_handle_p),
        cublas_handle(cublas_handle_p),
        data_tensor_desc(data_tensor_desc_p),

        conv1(cudnn_handle_p, cublas_handle_p, data_tensor_desc_p, 96, 11, 4, 0),
        relu1(cudnn_handle_p, conv1.output_tensor_desc, CUDNN_ACTIVATION_RELU),
        pool1(cudnn_handle_p, relu1.output_tensor_desc, 3, 2),

        conv2(cudnn_handle_p, cublas_handle_p, pool1.output_tensor_desc, 256, 5, 1, 0),
        relu2(cudnn_handle_p, conv2.output_tensor_desc, CUDNN_ACTIVATION_RELU),
        pool2(cudnn_handle_p, relu2.output_tensor_desc, 3, 2),

        conv3(cudnn_handle_p, cublas_handle_p, pool2.output_tensor_desc, 384, 3, 1, 0),
        relu3(cudnn_handle_p, conv3.output_tensor_desc, CUDNN_ACTIVATION_RELU),

        conv4(cudnn_handle_p, cublas_handle_p, relu3.output_tensor_desc, 384, 3, 1, 0),
        relu4(cudnn_handle_p, conv4.output_tensor_desc, CUDNN_ACTIVATION_RELU),

        conv5(cudnn_handle_p, cublas_handle_p, relu4.output_tensor_desc, 256, 3, 1, 0),
        relu5(cudnn_handle_p, conv5.output_tensor_desc, CUDNN_ACTIVATION_RELU),
        pool5(cudnn_handle_p, relu5.output_tensor_desc, 3, 2),

        fc6(cublas_handle_p, pool5.output_tensor_desc, 1024),
        relu6(cudnn_handle_p, fc6.output_tensor_desc, CUDNN_ACTIVATION_RELU),
        //drop 6

        fc7(cublas_handle_p, relu6.output_tensor_desc, 1024),
        relu7(cudnn_handle_p, fc7.output_tensor_desc, CUDNN_ACTIVATION_RELU),
        //drop 7

        fc8(cublas_handle_p, relu7.output_tensor_desc, 91),
        sm(cudnn_handle_p, fc8.output_tensor_desc),
        nll(cudnn_handle_p, sm.output_tensor_desc),

        gen(seed == 0 ? rd() : seed)
{
    conv1.init_weights_random(gen);
    conv2.init_weights_random(gen);
    conv3.init_weights_random(gen);
    conv4.init_weights_random(gen);
    conv5.init_weights_random(gen);
    fc6.init_weights_random(gen);
    fc7.init_weights_random(gen);
    fc8.init_weights_random(gen);
}


void ConvNet::fit(TrainData& train, int epoches, float lr){

    float epoch_loss;

    for (uint ep = 0; ep < epoches; ++ep) {
        epoch_loss = 0.0f;
        while (!train.is_finished()) {
            //std::cout << "Propagating next batch: " << train.get_n_read() << std::endl;

            train.load_next_batch();
            train.copy_batch_to_GPU();

            conv1.propagate_forward(train.d_img_data);
            relu1.propagate_forward(conv1.d_output);
            pool1.propagate_forward(relu1.d_output);

            conv2.propagate_forward(pool1.d_output);
            relu2.propagate_forward(conv2.d_output);
            pool2.propagate_forward(relu2.d_output);

            conv3.propagate_forward(pool2.d_output);
            relu3.propagate_forward(conv3.d_output);

            conv4.propagate_forward(relu3.d_output);
            relu4.propagate_forward(conv4.d_output);

            conv5.propagate_forward(relu4.d_output);
            relu5.propagate_forward(conv5.d_output);
            pool5.propagate_forward(relu5.d_output);

            fc6.propagate_forward(pool5.d_output);
            relu6.propagate_forward(fc6.d_output);
            fc7.propagate_forward(relu6.d_output);
            relu7.propagate_forward(fc7.d_output);
            fc8.propagate_forward(relu7.d_output);

            sm.propagate_forward(fc8.d_output);
            nll.propagate_forward(train.d_lbl_data, sm.d_output);


            epoch_loss += nll.batch_loss;


            nll.propagate_backward(train.d_lbl_data, sm.d_output);
            sm.propagate_backward(nll.d_dx, fc8.d_output);

            fc8.propagate_backward(sm.d_dx, relu7.d_output);
            relu7.propagate_backward(fc8.d_dx, fc7.d_output);
            fc7.propagate_backward(relu7.d_dx, relu6.d_output);
            relu6.propagate_backward(fc7.d_dx, fc6.d_output);
            fc6.propagate_backward(relu6.d_dx, pool5.d_output);

            pool5.propagate_backward(fc6.d_dx, relu5.d_output);
            relu5.propagate_backward(pool5.d_dx, conv5.d_output);
            conv5.propagate_backward(relu5.d_dx, relu4.d_output);

            relu4.propagate_backward(conv5.d_dx, conv4.d_output);
            conv4.propagate_backward(relu4.d_dx, relu3.d_output);

            relu3.propagate_backward(conv4.d_dx, conv3.d_output);
            conv3.propagate_backward(relu3.d_dx, pool2.d_output);

            pool2.propagate_backward(conv3.d_dx, relu2.d_output);
            relu2.propagate_backward(pool2.d_dx, conv2.d_output);
            conv2.propagate_backward(relu2.d_dx, pool1.d_output);

            pool1.propagate_backward(conv2.d_dx, relu1.d_output);
            relu1.propagate_backward(pool1.d_dx, conv1.d_output);
            conv1.propagate_backward(relu1.d_dx, train.d_img_data);


            conv1.update_weights(lr);
            conv2.update_weights(lr);
            conv3.update_weights(lr);
            conv4.update_weights(lr);
            conv5.update_weights(lr);
            fc6.update_weights(lr);
            fc7.update_weights(lr);
            fc8.update_weights(lr);

        }
        std::cout << "Epoch: " << ep
                  << "    Loss:" << epoch_loss
                  << std::endl;
        train.reset();
    }

}


int* ConvNet::predict(TestData& test){
    while (!test.is_finished()) {
        //std::cout << "Propagating next batch: " << train.get_n_read() << std::endl;

        test.load_next_batch();
        test.copy_batch_to_GPU();

        conv1.propagate_forward(test.d_img_data);
        relu1.propagate_forward(conv1.d_output);
        pool1.propagate_forward(relu1.d_output);

        conv2.propagate_forward(pool1.d_output);
        relu2.propagate_forward(conv2.d_output);
        pool2.propagate_forward(relu2.d_output);

        conv3.propagate_forward(pool2.d_output);
        relu3.propagate_forward(conv3.d_output);

        conv4.propagate_forward(relu3.d_output);
        relu4.propagate_forward(conv4.d_output);

        conv5.propagate_forward(relu4.d_output);
        relu5.propagate_forward(conv5.d_output);
        pool5.propagate_forward(relu5.d_output);

        fc6.propagate_forward(pool5.d_output);
        relu6.propagate_forward(fc6.d_output);
        fc7.propagate_forward(relu6.d_output);
        relu7.propagate_forward(fc7.d_output);
        fc8.propagate_forward(relu7.d_output);

        sm.propagate_forward(fc8.d_output);

        test.predict_batch_classes(sm.d_output);
    }
}
