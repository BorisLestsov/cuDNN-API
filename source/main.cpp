#include <iostream>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas.h>

#include "helper_functions.h"
#include "TrainData.h"
#include "Batch.h"

int main(){
	/*cudnnHandle_t handle;

    checkCUDNN( cudnnCreate(&handle) );

    checkCUDNN( cudnnDestroy(handle) );
*/

    TrainData train("dataset/imgdata.dat",
              "dataset/lbldata.dat",
                2, 256, 256, 3);

    Batch b = train.get_next_batch();


	return 0;
}