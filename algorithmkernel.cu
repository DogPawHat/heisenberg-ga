#include <cuda.h>
#include <math.h>
#include "ga_struts.h"

__device__ float * selection(int * matingPool, int * islandPoplulation, deviceFields fields, fieldSizes sizes){
	__shared__ float fitnessValues[sizes.populationSize];
	__shared__ float totalFinessValue;
	float selectedChromosome[sizes.chromosomeSize];
	int start = islandPoplulation[threadIdx.x*sizes.chromosoneSize];
	for(int i = 1; i < sizes.chromosoneSize; i++){
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

	for(int i = 0; i < sizes.poplulationSize; i++){
		if(diff < fitnessValues[i] - rouletteBall){
			fdif(fitnessValues[threadIdx.x], rouletteBall);
			memcpy(selectedChromosome, &islandPopulation[i*chromosomeSize], sizeof(int)*chromosomeSize);
		}
		__syncthreads();
	}
	return selectedIndividual;
}

__device__ void generation(int * islandPopulation, deviceFields fields, fieldSizes sizes){
	__shared__ float matingPool[sizes.populationSize/10];
	memcpy(&matingPool[threadIdx.x*chromosoneSize], selection(matingPool, islandPopulation, fields, sizes), sizeof(float)*chromosomeSize);
	__syncthreads();
	for(int i=0; i < popMultiplier; i++){
		memcpy(&islandPopulation[(threadIdx*popMultiplier+i)*chromosomeSize], &matingPool[threadIdx.x*chromosoneSize], chromosomesize*sizeof(int));
	}
}


__global__ void runGeneticAlgorithm(deviceFields fields, fieldSizes sizes){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int * islandPopulation = (int*) malloc(sizes.populationSizes)*sizes.chromosoneSize)];
	memcpy(&islandPopulation[threadIdx*chromosomeSize], &poplulation[gridIndex*chromosomeSize], chromosoneSize*sizeof(int));
	generation(islandPopulation, fields, sizes);
	memcpy(&poplulation[gridIndex*chromosomeSize], &islandPoplulation[blockDim.x*chromosomeSize], chromosomeSize*sizeof(int));
}
