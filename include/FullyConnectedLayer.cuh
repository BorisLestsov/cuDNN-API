#ifndef CUDNN_PROJ_FULLYCONNECTEDLAYER_H
#define CUDNN_PROJ_FULLYCONNECTEDLAYER_H

#include "Layer.cuh"
#include "cstdlib"

class FullyConnectedLayer: public Layer {

public:

    size_t n_inp, n_outp;

    float* weights, *bias;
    float* d_weights, *d_bias;


    FullyConnectedLayer(cublasHandle_t& cublas_handle_p, size_t n_inp_p, size_t n_outp_p);
    ~FullyConnectedLayer();


    void init_weights_random(/* rand function?*/);
    void load_weights_from_file(const char* fname);

    // TODO: ???
    void propagate_forward(/* ... */);

private:
    cublasHandle_t& cublas_handle;

};


#endif //CUDNN_PROJ_FULLYCONNECTEDLAYER_H
