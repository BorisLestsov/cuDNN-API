#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>

#include "helper_functions.h"


int main(){
	cudnnHandle_t handle;

    checkCUDNN( cudnnCreate(&handle) );

    checkCUDNN( cudnnDestroy(handle) );

	return 0;
}