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
                    "dataset/nmdata.dat",
                    "dataset/lbldata.dat",
                    3);

    while (!train.is_finished()){
        Batch b = train.get_next_batch();
        for (uint i = 0; i < b.size; ++i){
            std::cout << b.ids[i] << "   " << (int)  b.labels[i] << std::endl;
        }
        std::cout << std::endl;
    }


	return 0;
}