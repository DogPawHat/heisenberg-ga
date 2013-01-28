#include <cuda.h>
#include <math.h>

__device__ void selection(int * poplulation, float * tspGraph, const int chromosoneSize, const int popLength, const int graphSize){
	__device__ float * fitnessValues = (float *) malloc(sizeof(float) * popLength);
	int start = poplulation[(threadIdx.x + blockDim.x * blockIdx.x)*chromosoneSize];
	for(int i = 1; i < chromosoneSize; i++){
		float xd = tspGraph[start+(i*2)] - tspGraph[start+(i*2)-2];
		float yd = tspGraph[start+(i*2)+1] - tspGraph[start+(i*2)-1];
		fitnessValues[threadIdx.x + blockDim.x * blockIdx.x] += sqrt(xd^2 + yd^2);
	}
}

__device__ void generation(int * poplulation, float * tspGraph, const int chromosoneSize, const int popLength, const int graphSize){
	__shared__ int matingPool[blockDim.x*chromosoneSize];
	selection(mating);
	for(int j = 0; j < chromosoneSize; j++){
		poplulation[threadIdx.x + blockDim.x*blockIdx.x + i] = matingPool[blockDim.x + i];
	}
}


__global__ void runGeneticAlgorithm(int * poplulation, float * tspGraph, const int chromosoneSize, const int popLength, const int graphSize){
	__shared__ int matingPool[blockDim.x*chromosoneSize];
	for(int i = 0; i < chromosoneSize; i++){
		matingPool[threadIdx + i] = poplulation[threadIdx.x + blockDim.x*blockIdx.x + i];
	}
	generation(matingPool, poplulation, tspGraph, popLength, graphSize);
	for(int j = 0; j < chromosoneSize; j++){
		poplulation[threadIdx.x + blockDim.x*blockIdx.x + i] = matingPool[blockDim.x + i];
	}
}