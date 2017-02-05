#include "FullyConnectedLayer.h"


FullyConnectedLayer::FullyConnectedLayer(size_t n_inp_p, size_t n_outp_p):
        n_inp(n_inp_p),
        n_outp(n_outp_p)
{
    weights = (float*) malloc(n_inp * n_outp);
    bias = (float*) malloc(n_outp);
}


FullyConnectedLayer::~FullyConnectedLayer() {
    free(weights);
    free(bias);
}
