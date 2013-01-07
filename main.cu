#include "cuda.h"
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "rand.h"

int main(){
	int test[52];
	int testHost[52];
	int * testDevice;
	curandState* devStates;
	cudaMalloc(&devStates, sizeof(curandState) );
	cudaMalloc(&testDevice, sizeof(int)*52);
	testRand<<<1, 1>>>(testDevice, devStates);
	cudaMemcpy(testHost, testDevice, sizeof(testDevice),cudaMemcpyDeviceToHost);
	for (int i = 0; i < 52; i++){
		std::cout << test[i] << std::endl;
	}
	std::cout << "Press any key to exit";
	std::cin >> new char;
}
