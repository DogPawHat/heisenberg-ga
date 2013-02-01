#include <cuda.h>
#include <math.h>

__device__ float * selection(int * matingPool, int * islandPoplulation, float * tspGraph, const int chromosoneSize, const int islandPopLength, const int graphSize){
	__shared__ float fitnessValues[popLength];
	__shared__ float totalFinessValue;
	float selectedChromosome[chromosomeSize];
	int start = islandPoplulation[threadIdx.x*chromosoneSize];
	for(int i = 1; i < chromosoneSize; i++){
		float xd = tspGraph[start+(i*2)] - tspGraph[start+(i*2)-2];
		float yd = tspGraph[start+(i*2)+1] - tspGraph[start+(i*2)-1];
		fitnessValues[threadIdx.x] += sqrt(xd^2 + yd^2);
		totalFitnessValue += sqrt(xd^2 + yd^2);
		__syncthreads();
	}
	fitnessValues[threadIdx.x] = finessValues[threadIdx.x]/totalFitnessValue;
	__syncthreads();

	float rouletteBall = randomrouletteBall();
	unsigned float diff = fdif(fitnessValues[threadIdx.x], rouletteBall);
	memcpy(selectedIndividual, &islandPopulation[threadIdx.x*chromosomeSize], sizeof(int)*chromosomeSize);

	for(int i = 0; i < popLength; i++){
		if(diff < fitnessValues[i] - rouletteBall){
			fdif(fitnessValues[threadIdx.x], rouletteBall);
			memcpy(selectedChromosome, &islandPopulation[i*chromosomeSize], sizeof(int)*chromosomeSize);
		}
		__syncthreads();
	}
	return selectedIndividual;
}

__device__ void generation(islandPopulation, poplulation, tspGraph, chromosoneSize, popLength, graphSize, popMultipler){
	__shared__ float matingPool[blockDim.x*chromosoneSize];
	memcpy(&matingPool[threadIdx.x*chromosoneSize], selection(matingPool, islandPopulation, tspGraph, chromosoneSize, popLength/gridDim.x, graphSize), sizeof(float)*chromosomeSize);
	__syncthreads();
	for(int i=0; i < popMultiplier; i++){
		memcpy(&islandPopulation[(threadIdx*popMultiplier+i)*chromosomeSize], &matingPool[threadIdx.x*chromosoneSize], chromosomesize*sizeof(int));
	}
}


__global__ void runGeneticAlgorithm(int * poplulation, float * tspGraph, const int chromosoneSize, const int popLength, const int graphSize, const int popMultipler){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int * islandPopulation = new int[(popLength/gridDim.x)*chromosoneSize)];
	for(int i=0; i < popMultiplier; i++){
		memcpy(&islandPopulation[(threadIdx*popMultiplier+i)*chromosomeSize], &poplulation[(gridIndex*popMultiplier+i)*chromosomeSize], chromosomesize*sizeof(int));
	}
	generation(islandPopulation, poplulation, tspGraph, chromosoneSize, popLength, graphSize, popMultipler);
	for(int j = 0; j < chromosoneSize; j++){
		poplulation[threadIdx.x + blockDim.x*blockIdx.x + i] = islandPoplulation[blockDim.x + i];
	}
}
