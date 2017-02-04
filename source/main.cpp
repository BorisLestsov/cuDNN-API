#include "TrainData.h"
#include "ConvNet.cuh"

#include <iostream>


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
        train.load_next_batch();
        for (uint i = 0; i < train.loaded; ++i){
            std::cout << train.ids_data[i] << "   " << (int) train.lbl_data[i] << std::endl;
        }
        std::cout << std::endl;
    }

    ConvNet alexnet;
    alexnet.fit(train);

	return 0;
}