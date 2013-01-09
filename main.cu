#include "cuda.h"
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "rand.h"



int main(){
	const int testSize = 52;
	float testHost[testSize];
	float * testDevice;
	cudaMalloc((void**)&testDevice, testSize*sizeof(float));
	testRand<<<1, 1>>>(testDevice, testSize);
	cudaMemcpy(testHost, testDevice, sizeof(float)*testSize,cudaMemcpyDeviceToHost);
	for (int i = 0; i < 52; i++){
		std::cout << testHost[i] << std::endl;
	}
	cudaFree(testDevice);
	std::cin.get();
}
