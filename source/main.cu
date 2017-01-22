#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>

#include "helper_functions.h"

__global__ void f() {

}

int main(){
	cudnnHandle_t handle;

    f<<<1, 1>>>();

    checkCUDNN( cudnnCreate(&handle) );

    checkCUDNN( cudnnDestroy(handle) );

	return 0;
}