#include "TrainData.cuh"
#include "ConvNet.cuh"

#include <iostream>


int main(){
    init_cuda();

    cudnnHandle_t handle;
    checkCUDNN( cudnnCreate(&handle) );

    TrainData train(handle,
                    "dataset/imgdata.dat",
                    "dataset/nmdata.dat",
                    "dataset/lbldata.dat",
                    2);

    while (!train.is_finished()){
        train.load_next_batch();
        for (uint i = 0; i < train.loaded; ++i){
            std::cout << train.ids_data[i] << "   " << train.lbl_data[i] << std::endl;
        }
        std::cout << std::endl;
    }

    ConvNet alexnet;
    alexnet.fit(train);


    checkCUDNN( cudnnDestroy(handle) );
    return 0;
}