#include "ConvNet.hpp"


ConvNet::ConvNet(cudnnHandle_t& cudnn_handle_p,
                 cublasHandle_t& cublas_handle_p,
                 cudnnTensorDescriptor_t data_tensor_desc_p,
                 uint seed):

        metric(nullptr),
        cudnn_handle(cudnn_handle_p),
        cublas_handle(cublas_handle_p),
        data_tensor_desc(data_tensor_desc_p)
{}


void ConvNet::add_layer(Layer* layer){
    layers.push_back(layer);
}

cudnnTensorDescriptor_t ConvNet::last_layer_outp_desc(){
    return layers.back()->output_tensor_desc;
}

void ConvNet::set_metric(MetricLayer* metric_p){
    metric = metric_p;
}

void ConvNet::fit(TrainData& train, size_t epoches, float lr, float momentum){

    float epoch_loss;

    for (size_t ep = 0; ep < epoches; ++ep) {
        epoch_loss = 0.0f;
        while (!train.is_finished()) {
            //std::cout << "Propagating next batch: " << train.get_n_read() << std::endl;

            train.load_next_batch();
            train.copy_batch_to_GPU();

            layers[0]->propagate_forward(train.d_img_data);
            for(size_t i = 1; i < layers.size(); ++i){
                layers[i]->propagate_forward(layers[i-1]->d_output);
            }
            metric->compute_loss(train.d_lbl_data, layers.back()->d_output);

            epoch_loss += metric->batch_loss;
            //std::cout << "Batch Loss: " << metric->batch_loss << std::endl;

            metric->propagate_backward(train.d_lbl_data, layers.back()->d_output);
            layers.end()[-1]->propagate_backward(metric->d_dx, layers.end()[-2]->d_output, momentum);
            for(long i = layers.size()-2; i >= 1; --i){
                layers[i]->propagate_backward(layers[i+1]->d_dx, layers[i-1]->d_output, momentum);
            }
            layers.front()->propagate_backward(layers[1]->d_dx, train.d_img_data, momentum);

            for(size_t i = 0; i < layers.size(); ++i) {
                layers[i]->update_weights(lr);
            }
        }

        std::cout << "Epoch: " << ep
                  << "    Loss:" << epoch_loss
                  << std::endl;
        train.reset();
    }

}


std::vector<int> ConvNet::predict_labels(TestData& test){
    std::vector<int> res;
    std::vector<int> tmp;

    test.reset();
    while (!test.is_finished()) {
        //std::cout << "Propagating next batch: " << train.get_n_read() << std::endl;

        test.load_next_batch();
        test.copy_batch_to_GPU();

        layers[0]->propagate_forward(test.d_img_data);
        for(size_t i = 1; i < layers.size(); ++i){
            layers[i]->propagate_forward(layers[i-1]->d_output);
        }

        tmp = test.predict_batch_classes(layers.end()[-1]->d_output);
        res.insert(res.end(), tmp.begin(), tmp.end());
    }

    return res;
}
