#ifndef CUDNN_PROJ_CONVNET_CUH
#define CUDNN_PROJ_CONVNET_CUH

#include "helper_functions.cuh"
#include "TrainData.cuh"
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


#endif //CUDNN_PROJ_CONVNET_CUH
