#ifndef CUDNN_PROJ_FULLYCONNECTEDLAYER_H
#define CUDNN_PROJ_FULLYCONNECTEDLAYER_H

#include "Layer.h"
#include "cstdlib"

class FullyConnectedLayer: public Layer {

public:

    size_t n_inp, n_outp;

    float* weights, *bias;
    float* d_weights, *d_bias;


    FullyConnectedLayer(size_t n_inp_p, size_t n_outp_p);
    ~FullyConnectedLayer();


    void init_weights_random(/* rand function?*/);
    void load_weights_from_file(const char* fname);

    // TODO: ???
    void propagate_forward(/* ... */);

private:


};


#endif //CUDNN_PROJ_FULLYCONNECTEDLAYER_H
