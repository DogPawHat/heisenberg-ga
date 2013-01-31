#include <cuda.h>
#include <math.h>

__device__ void selection(int * poplulation, float * tspGraph, const int chromosoneSize, const int popLength, const int graphSize){
	__shared__ float fitnessValues[popLength];
	int start = poplulation[(threadIdx.x + blockDim.x * blockIdx.x)*chromosoneSize];
	for(int i = 1; i < chromosoneSize; i++){
		float xd = tspGraph[start+(i*2)] - tspGraph[start+(i*2)-2];
		float yd = tspGraph[start+(i*2)+1] - tspGraph[start+(i*2)-1];
		fitnessValues[threadIdx.x + blockDim.x * blockIdx.x] += sqrt(xd^2 + yd^2);
	}
}

__device__ void generation(islandPopulation, poplulation, tspGraph, chromosoneSize, popLength, graphSize, popMultipler){
	__shared__ int matingPool[blockDim.x*chromosoneSize];
	selection(matingPool, islandPopulation, poplulation, tspGraph, chromosoneSize, popLength, graphSize, popMultipler);
	for(int j = 0; j < chromosoneSize; j++){
		poplulation[threadIdx.x + blockDim.x*blockIdx.x + i] = matingPool[blockDim.x + i];
	}
}


__global__ void runGeneticAlgorithm(int * poplulation, float * tspGraph, const int chromosoneSize, const int popLength, const int graphSize, const int popMultipler){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int * islandPopulation = new int[(popLength/gridDim.x)*chromosoneSize)];
	for(int i=0; i < popMultiplier; i++){
		islandPopulation[(threadIdx*popMultiplier+i)*chromosomeSize] = poplulation[(gridIndex*popMultiplier+i)*chromosomeSize];
	}
	generation(islandPopulation, poplulation, tspGraph, chromosoneSize, popLength, graphSize, popMultipler);
	for(int j = 0; j < chromosoneSize; j++){
		poplulation[threadIdx.x + blockDim.x*blockIdx.x + i] = matingPool[blockDim.x + i];
	}
}