#include <cuda.h>
#include <math.h>

__device__ void selection(int * matingPool, int * islandPoplulation, float * tspGraph, const int chromosoneSize, const int islandPopLength, const int graphSize){
	__shared__ float fitnessValues[popLength];
	__shared__ float totalFinessValue;
	int start = islandPoplulation[threadIdx.x*chromosoneSize];
	for(int i = 1; i < chromosoneSize; i++){
		float xd = tspGraph[start+(i*2)] - tspGraph[start+(i*2)-2];
		float yd = tspGraph[start+(i*2)+1] - tspGraph[start+(i*2)-1];
		fitnessValues[threadIdx.x] += sqrt(xd^2 + yd^2);
		totalFitnessValue += sqrt(xd^2 + yd^2);
	}
	fitnessValues[threadIdx.x] = finessValues[threadIdx.x]/totalFitnessValue;
	
	
	
}

__device__ void generation(islandPopulation, poplulation, tspGraph, chromosoneSize, popLength, graphSize, popMultipler){
	__shared__ int matingPool[blockDim.x*chromosoneSize];
	selection(matingPool, islandPopulation, tspGraph, chromosoneSize, popLength/gridDim.x, graphSize);
}


__global__ void runGeneticAlgorithm(int * poplulation, float * tspGraph, const int chromosoneSize, const int popLength, const int graphSize, const int popMultipler){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int * islandPopulation = new int[(popLength/gridDim.x)*chromosoneSize)];
	for(int i=0; i < popMultiplier; i++){
		memcpy(&islandPopulation[(threadIdx*popMultiplier+i)*chromosomeSize], &poplulation[(gridIndex*popMultiplier+i)*chromosomeSize], chromosomesize*sizeof(int));
	}
	generation(islandPopulation, poplulation, tspGraph, chromosoneSize, popLength, graphSize, popMultipler);
	for(int j = 0; j < chromosoneSize; j++){
		poplulation[threadIdx.x + blockDim.x*blockIdx.x + i] = matingPool[blockDim.x + i];
	}
}
