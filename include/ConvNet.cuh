#ifndef CUDNN_PROJ_CONVNET_H
#define CUDNN_PROJ_CONVNET_H

#include "helper_functions.h"
#include "TrainData.h"
#include "TestData.h"

#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>


class ConvNet {
public:

    void fit(TrainData&);
    char* predict(TestData&);

private:
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;
};


#endif //CUDNN_PROJ_CONVNET_H
