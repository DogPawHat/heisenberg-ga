#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include "global_structs.h"


__device__ void crossover(metaChromosome[], deviceFields);
__device__ void selection(metaChromosome[], deviceFields);
__device__ void mutation(metaChromosome[], deviceFields);
__device__ void createNewSeed(deviceFields, long);
__device__ void fitnessEvauation(metaChromosome[], deviceFields);
__device__ void bitonicSort(metaChromosome islandPopulation[]);


__device__ void generation(metaChromosome islandPopulation[], deviceFields fields){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	islandPopulation[threadIdx.x] = fields.population[gridIndex];
	__syncthreads();

//	mutation(islandPopulation, fields);
//	__syncthreads();

	crossover(islandPopulation, fields);
	__syncthreads();

	selection(islandPopulation, fields);
	__syncthreads();

/*	if(blockIdx.x <= (GRID_SIZE - 2) && threadIdx.x >= BLOCK_SIZE/2){
		fields.population[gridIndex + ISLAND_POPULATION_SIZE] = islandPopulation[threadIdx.x];
	}else if(blockIdx.x == (GRID_SIZE - 1) && threadIdx.x >= BLOCK_SIZE/2){
		fields.population[threadIdx.x] = islandPopulation[threadIdx.x];
	}else if(threadIdx.x <= BLOCK_SIZE/2){
		fields.population[gridIndex] = islandPopulation[threadIdx.x];
	}
*/

	fields.population[gridIndex] = islandPopulation[threadIdx.x];
	__syncthreads();
}


__global__ void runGeneticAlgorithm(deviceFields fields){
	__shared__ metaChromosome islandPopulation[ISLAND_POPULATION_SIZE];
	for(int i = 0; i < GENERATIONS; i++){
		generation(islandPopulation, fields);
		__syncthreads();
	}
}



/*Selection Functions*/

__device__ float randomRouletteBall(deviceFields fields){
	thrust::minstd_rand0 rng(fields.seeds[threadIdx.x + blockDim.x*blockIdx.x]);
	thrust::uniform_real_distribution<float> dist(0, 1);
	float result = dist(rng);
	return result;
}

__device__ void selection(metaChromosome islandPopulation[], deviceFields fields){
	__shared__ metaChromosome selectedPopulation[ISLAND_POPULATION_SIZE];
	float rouletteBall;
	float currentFitnessInterval;

	islandPopulation[threadIdx.x].distanceCalculation(fields.TSPGraph);
	__syncthreads();
	fitnessEvauation(islandPopulation, fields);
	__syncthreads();

	rouletteBall = randomRouletteBall(fields);
	currentFitnessInterval = islandPopulation[0].fitness;
	selectedPopulation[threadIdx.x] = islandPopulation[0];

	for(short i = 1; i < ISLAND_POPULATION_SIZE; i++){
		if(rouletteBall > currentFitnessInterval){
			currentFitnessInterval += islandPopulation[i].fitness;
		}else{
			selectedPopulation[threadIdx.x] = islandPopulation[i];
			break;
		}
	}

	islandPopulation[threadIdx.x] = selectedPopulation[threadIdx.x];
	__syncthreads();
}

__device__ void fitnessEvauation(metaChromosome islandPopulation[], deviceFields fields){
	__shared__ float sumOfFitnessValues[ISLAND_POPULATION_SIZE];
	islandPopulation[threadIdx.x].fitness = 1/islandPopulation[threadIdx.x].distance;
	sumOfFitnessValues[threadIdx.x] = islandPopulation[threadIdx.x].fitness;
	__syncthreads();

	for(short stride = 1; stride < ISLAND_POPULATION_SIZE; stride *= 2){
		if(threadIdx.x + stride < ISLAND_POPULATION_SIZE){
			sumOfFitnessValues[threadIdx.x] += sumOfFitnessValues[threadIdx.x+stride];
			__syncthreads();
		}
	}

	islandPopulation[threadIdx.x].fitness = islandPopulation[threadIdx.x].fitness/sumOfFitnessValues[0];
}

__device__ void bitonicStep(int stride, metaChromosome islandPopulation[]){
	metaChromosome temp;
	if((threadIdx.x % stride) >= 0 && (threadIdx.x % stride) < stride/2){
		if((threadIdx.x/stride)%2 == 0){
			if(islandPopulation[threadIdx.x].fitness > islandPopulation[threadIdx.x + (stride/2)].fitness){
				temp = islandPopulation[threadIdx.x];
				islandPopulation[threadIdx.x] = islandPopulation[threadIdx.x + (stride/2)];
				islandPopulation[threadIdx.x + (stride/2)] = temp;
			}
		}else{
			if(islandPopulation[threadIdx.x].fitness < islandPopulation[threadIdx.x + (stride/2)].fitness){
				temp = islandPopulation[threadIdx.x + (stride/2)];
				islandPopulation[threadIdx.x + (stride/2)] = islandPopulation[threadIdx.x];
				islandPopulation[threadIdx.x] = temp;
			}
		}
	}
}

__device__ void bitonicSort(metaChromosome islandPopulation[]){
	for(int i = 2; i <= ISLAND_POPULATION_SIZE; i*2){
		for(int j = i; j < 1; j/2){
			bitonicStep(j, islandPopulation);
			__syncthreads();
		}
	}
}


/*Genetic Operators*/

__device__ short pmxSwap(short value, metaChromosome parent1, short point1, metaChromosome parent2, short point2){
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		if(value == parent2.chromosome[i]){
			if(i < point1 || i > point2){
				return i;
			}else{
				return pmxSwap(parent1.chromosome[i], parent1, point1, parent2, point2);
				}
			}
		}
	return 0;
}

__device__ void crossover(metaChromosome islandPopulation[], deviceFields fields){
	/*We need two different paths here beause each thread needs two parents to generate a single offspring.
	The first half of the block will take one parent from the first half of islandPopulation, while the second parent
	will come from the second half. This is reversed for the second half of the block. To reduce warp control divergence,
	block size shoud be a multiple of 2*warp size, 32 being the current value of warps in Fermi and Kepler GPU's*/
	
	metaChromosome parent1; //Points to the first element in the chromosome of parent1
	metaChromosome parent2;
	short point1;
	short point2;
	metaChromosome offspring = islandPopulation[threadIdx.x];
	thrust::minstd_rand0 rng(fields.seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	thrust::uniform_int_distribution<short> dist1;
	thrust::uniform_int_distribution<short> dist2;

	if(threadIdx.x < (BLOCK_SIZE/2)){
		parent1 = islandPopulation[threadIdx.x];
		parent2 = islandPopulation[threadIdx.x+(BLOCK_SIZE/2)];
	}else{
		parent1 = islandPopulation[threadIdx.x];
		parent2 = islandPopulation[threadIdx.x-(BLOCK_SIZE/2)];
	}

	dist1 = thrust::uniform_int_distribution<short>(0, CHROMOSOME_SIZE-1);
	point1 = dist1(rng);
	dist2 = thrust::uniform_int_distribution<short>(point1, CHROMOSOME_SIZE-1);
	point2 = dist2(rng);

	offspring = parent2;

	for(short i = point1; i <= point2; i++){
		offspring.chromosome[i] = parent1.chromosome[i];
	}

	for(short i = point1; i <= point2; i++){
		if(offspring.chromosome[i] != parent2.chromosome[i]){
			offspring.chromosome[pmxSwap(parent1.chromosome[i], parent1, point1, parent2, point2)] = parent2.chromosome[i];
		}
	}

	offspring.distanceCalculation(fields.TSPGraph);
	islandPopulation[threadIdx.x] = offspring;
}

__device__ void mutation(metaChromosome islandPopulation[], deviceFields fields){
	metaChromosome mutant = islandPopulation[threadIdx.x]; 
	thrust::minstd_rand0 rng(fields.seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	thrust::uniform_int_distribution<short> dist1(0, 10);
	thrust::uniform_int_distribution<short> dist2(0, 51);
	short numOfSwaps = dist1(rng);
	short swapPoint1;
	short swapPoint2;
	short temp;

	for(int i = 0; i < numOfSwaps; i++){
		swapPoint1 = dist2(rng);
		swapPoint2 = dist2(rng);
		temp = mutant.chromosome[swapPoint1];
		mutant.chromosome[swapPoint1] = mutant.chromosome[swapPoint2];
		mutant.chromosome[swapPoint2] = temp;
	}

	islandPopulation[threadIdx.x] = mutant;
}

__device__ void createNewSeed(deviceFields fields, long seed){
	thrust::minstd_rand0 rng(seed);

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	fields.seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


