#include "cuda.h"
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "berlin52.h"
#include "rand.h"
#include "ga_struts.h"

#define BLOCK_SIZE 32
#define GRID_SIZE 10

int main(){
	const fieldSizes sizes = {BLOCK_SIZE*GRID_SIZE, 52};
	hostFields host;
	deviceFields device;

	cudaMallocHost((void**) &host.population, sizes.populationSize*sizeof(int));
	cudaMalloc((void**) &device.source, sizes.chromosomeSize*sizeof(int));
	cudaMalloc((void**) &device.population, sizes.populationSize*sizeof(int));
	cudaMalloc((void**) &device.TSPGraph, sizes.chromosomeSize*2*sizeof(float));
	cudaMemcpy(device.TSPGraph, berlin52, sizes.chromosomeSize*2*sizeof(float), cudaMemcpyHostToDevice);
	thrust::device_ptr<int> sourceThrust = thrust::device_pointer_cast(source);
	thrust::sequence(sourceThrust, sourceThrust + sizes.chromosomeSize);

	createRandomPermutation<<<GRID_SIZE, BLOCK_SIZE>>>(device, sizes, time(NULL));
	cudaDeviceSynchronize();

	cudaMemcpy(hostPopulation, devicePopulation, populationSize*sizeof(int),cudaMemcpyDeviceToHost);
	for (int i = 0; i < populationSize; i++){
		for(int j = 0; j < chromosomeSize; j++){
			std::cout << hostPopulation[j+i*chromosomeSize] << " ";
		}
		std::cout << std::endl;
	}

	cudaFree(devicePopulation);
	cudaFree(deviceTSPRoute);
	cudaFree(source);
	cudaHostFree(hostPopulation);
	std::cin.get();
}