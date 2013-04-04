#ifndef ALGORITHM_CUH
#define ALGORITHM_CUH

#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>


const int BLOCK_SIZE = 512;
const int GRID_SIZE = 12;

class metaChromosome;
class geneticAlgorithm;

__shared__ int * islandPopulationChromosome[BLOCK_SIZE];
__shared__ double islandPopulationDistance[BLOCK_SIZE];
__shared__ int warpCrossoverChance[BLOCK_SIZE/32];
__shared__ int warpMutationChance[BLOCK_SIZE/32];

class geneticAlgorithm{
public:
	const int GENERATIONS;
	const int CHROMOSOME_SIZE;
	const int POPULATION_SIZE;
	const int ISLAND_POPULATION_SIZE;
	const int CROSSOVER_CHANCE;
	const int MUTATION_CHANCE;
	const int OPTIMAL_LENGTH;
	int runGenerations;
	int * source;
	int * seeds;
	double * adjacencyMatrix;
	int * populationChromosome;
	double * populationDistance;
	bool optimalLengthReached[GRID_SIZE];


	__host__ __device__ geneticAlgorithm(const int GENERATIONS, const int OPTIMAL_LENGTH, const int CHROMOSOME_SIZE, const int CROSSOVER_CHANCE = 90, const int MUTATION_CHANCE = 25)
		: GENERATIONS(GENERATIONS),
		  OPTIMAL_LENGTH(OPTIMAL_LENGTH),
		CHROMOSOME_SIZE(CHROMOSOME_SIZE),
		POPULATION_SIZE(BLOCK_SIZE*GRID_SIZE),
		ISLAND_POPULATION_SIZE(BLOCK_SIZE),
		CROSSOVER_CHANCE(CROSSOVER_CHANCE),
		MUTATION_CHANCE(MUTATION_CHANCE) ,
		runGenerations(0)
	{
		for(int i = 0; i < GRID_SIZE; i++){
			optimalLengthReached[i] = false;
		}

	}

	__device__ void generation();
	__device__ void distanceCalculation();
	__device__ double distanceCalculation(int*);
	__device__ void migration();

private:


	__device__ void createNewSeed(long);

	__device__ void selection();
	__device__ void tournamentSelection();

	__device__ void sort();
	__device__ void exchange(int *, int *);

	__device__ void crossover();
	__device__ void crossoverPMX(int*, int*);

	__device__ void mutation();


	__host__ __device__ double distanceBetweenTwoCities(int, int);
};

__global__ void createRandomPermutation(geneticAlgorithm algorithm){
	int * tempResult = new int[algorithm.CHROMOSOME_SIZE];
	int temp;
	int rand;
	int * chromosome = &(algorithm.populationChromosome[(threadIdx.x+blockIdx.x*blockDim.x)*algorithm.CHROMOSOME_SIZE]);

	thrust::minstd_rand0 rng(algorithm.seeds[threadIdx.x+blockIdx.x*blockDim.x]);

	for(int i = 0; i < algorithm.CHROMOSOME_SIZE; i++){
		tempResult[i] = algorithm.source[i];
	}

	for(int i = algorithm.CHROMOSOME_SIZE-1; i >= 0; i--){
		thrust::uniform_int_distribution<int> dist(0,i);
		rand = dist(rng);
		temp = tempResult[rand];
		tempResult[rand] = tempResult[i];
		tempResult[i] = temp;
	}
	__syncthreads();

	for(int i = 0; i < algorithm.CHROMOSOME_SIZE; i++){
		chromosome[i] = tempResult[i];
	}
	algorithm.populationDistance[threadIdx.x+blockIdx.x*blockDim.x] = algorithm.distanceCalculation(chromosome);
	if(threadIdx.x == 0){
		delete tempResult;
	}
}

__global__ void createRandomSeeds(geneticAlgorithm algorithm, long seed){

	thrust::minstd_rand0 rng(seed*(threadIdx.x + blockIdx.x*blockDim.x));

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	algorithm.seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


__global__ void runOneGeneration(geneticAlgorithm algorithm){
	algorithm.generation();
}

__device__ void geneticAlgorithm::generation(){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	int crossoverChance;
	int mutationChance;

	
	islandPopulationChromosome[threadIdx.x] = &populationChromosome[gridIndex*CHROMOSOME_SIZE];


	distanceCalculation();
	__syncthreads();

	thrust::minstd_rand rng(seeds[gridIndex]);
	thrust::uniform_int_distribution<short> dist(1, 100);

	selection();
	__syncthreads();


	if(threadIdx.x % 32 == 0){
		warpCrossoverChance[threadIdx.x/32] = dist(rng);
		warpMutationChance[threadIdx.x/32] = dist(rng);
	}
	__syncthreads();

	createNewSeed(seeds[gridIndex]);
	__syncthreads();

	if(warpCrossoverChance[threadIdx.x/32] < CROSSOVER_CHANCE){
		crossover();
		__syncthreads();
	}

	createNewSeed(seeds[gridIndex]);
	__syncthreads();

	if(warpMutationChance[threadIdx.x/32] < MUTATION_CHANCE){
		mutation();
		__syncthreads();
	}

	distanceCalculation();
	__syncthreads();

	createNewSeed(seeds[gridIndex]);
	__syncthreads();

	sort();
	__syncthreads();
	
	distanceCalculation();
	__syncthreads();

	populationDistance[gridIndex] = islandPopulationDistance[threadIdx.x];
	__syncthreads();
}


__global__ void runOneMigration(geneticAlgorithm algorithm){
	algorithm.migration();
}

/* Migration Functions */

__device__ void geneticAlgorithm::migration(){
	int gridIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int divisor = 4;
	if(threadIdx.x >= ISLAND_POPULATION_SIZE/divisor){
		int migrationBlockOffset = threadIdx.x/(ISLAND_POPULATION_SIZE/divisor);
		int migrationSourceBlockIdx = (migrationBlockOffset+blockIdx.x)%GRID_SIZE;
		int migrationDestinationThreadIdx = gridIndex;
		int migrationSourceThreadIdx = migrationSourceBlockIdx*blockDim.x + (threadIdx.x%(ISLAND_POPULATION_SIZE/divisor));
		populationDistance[migrationDestinationThreadIdx] = populationDistance[migrationSourceThreadIdx];
		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			populationChromosome[migrationDestinationThreadIdx*CHROMOSOME_SIZE+i] = populationChromosome[migrationSourceThreadIdx*CHROMOSOME_SIZE+i];
		}
	}
	__syncthreads();
	if(populationDistance[gridIndex] < OPTIMAL_LENGTH && threadIdx.x == 0){
		optimalLengthReached[blockIdx.x] = true;
	}

}


/*Random Number Generator functions*/

__device__ void geneticAlgorithm::createNewSeed(long seed){
	thrust::minstd_rand rng(seed);

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


/*Selection Functions*/

__device__ void geneticAlgorithm::selection(){
	tournamentSelection();
}

__device__ void geneticAlgorithm::tournamentSelection(){
	int N = 5;
	int tournamentChampion;
	int tournamentChallenger;
	int temp;

	thrust::minstd_rand rng(seeds[threadIdx.x + blockIdx.x*blockDim.x]);
	thrust::uniform_int_distribution<int> dist(0, ISLAND_POPULATION_SIZE-1);
	
	tournamentChampion = threadIdx.x;

	for(int i = 0; i < N; i++){
		tournamentChallenger = dist(rng);
		if(islandPopulationDistance[tournamentChampion] > islandPopulationDistance[tournamentChallenger]){
			tournamentChampion = tournamentChallenger;
		}
	}
	__syncthreads();

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		temp = islandPopulationChromosome[tournamentChampion][i];
		__syncthreads();
		islandPopulationChromosome[threadIdx.x][i] = temp;
		__syncthreads();
	}
	distanceCalculation();
}

/* Sorting Algorithms */

__device__ void geneticAlgorithm::sort(){
	__shared__ int rank[BLOCK_SIZE];

	rank[threadIdx.x] = threadIdx.x;


	for (int k = 2; k <= ISLAND_POPULATION_SIZE; k <<= 1){
		__syncthreads();
		for (int j=k>>1; j>0; j=j>>1){
			__syncthreads();
			unsigned int i = threadIdx.x; 
			int ixj = i^j;

			if ((ixj)>i){
				if ((i&k)==0 && islandPopulationDistance[i]>islandPopulationDistance[ixj]){
					double distanceTemp = islandPopulationDistance[i];
					double rankTemp = rank[i];
					islandPopulationDistance[i] = islandPopulationDistance[ixj];
					rank[i] = rank[ixj];
					islandPopulationDistance[ixj] = distanceTemp;
					rank[ixj] = rankTemp;
				}
				if ((i&k)!=0 && islandPopulationDistance[i]<islandPopulationDistance[ixj]){
					double distanceTemp = islandPopulationDistance[i];
					double rankTemp = rank[i];
					islandPopulationDistance[i] = islandPopulationDistance[ixj];
					rank[i] = rank[ixj];
					islandPopulationDistance[ixj] = distanceTemp;
					rank[ixj] = rankTemp;
				}
			}
			__syncthreads();
		}
	}

	__syncthreads();
	int * currentChromosome = islandPopulationChromosome[threadIdx.x];
	int * newChromosome = islandPopulationChromosome[rank[threadIdx.x]];
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		int temp = newChromosome[i];
		__syncthreads();
		currentChromosome[i] = temp;
	}
}

__device__ void geneticAlgorithm::exchange(int * chromosome1, int * chromosome2){
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		int temp = chromosome1[i];
		__syncthreads();
		chromosome1[i] = chromosome2[i];
		__syncthreads();
		chromosome2[i] = temp;
		__syncthreads();
	}
}


/*Genetic Operators*/

__device__ void geneticAlgorithm::mutation(){
	int * mutant = new int[CHROMOSOME_SIZE];
	thrust::minstd_rand0 rng(seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	int swapPoint1;
	int swapPoint2;
	int temp;

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		mutant[i] = islandPopulationChromosome[threadIdx.x][i];
	}

	thrust::uniform_int_distribution<short> dist1(0, CHROMOSOME_SIZE-1);
	swapPoint1 = dist1(rng);
	thrust::uniform_int_distribution<short> dist2(swapPoint1, CHROMOSOME_SIZE-1);
	swapPoint2 = dist2(rng);
	int offset = swapPoint2 - swapPoint1;
	for(int i = swapPoint1; i <= swapPoint2; i++){
		temp = mutant[i];
		mutant[i] = mutant[swapPoint2-(i-swapPoint1)];
		mutant[swapPoint2-(i-swapPoint1)] = temp;
	}

	__syncthreads();
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulationChromosome[threadIdx.x][i] = mutant[i];
	}
	distanceCalculation();
	delete mutant;
}

__device__ void geneticAlgorithm::crossover(){
	int * parent1;
	int * parent2;
	thrust::minstd_rand rng(seeds[threadIdx.x + blockIdx.x*blockDim.x]);
	thrust::uniform_int_distribution<int> dist(0, ISLAND_POPULATION_SIZE-1);

	parent1 = islandPopulationChromosome[threadIdx.x];
	parent2 = islandPopulationChromosome[dist(rng)];

	crossoverPMX(parent1, parent2);
}

__device__ void geneticAlgorithm::crossoverPMX(int * parent1, int * parent2){
	/*We need two different paths here because each thread needs two parents to generate a single offspring.
	The first half of the block will take one parent from the first half of islandPopulation, while the second parent
	will come from the second half. This is reversed for the second half of the block. To reduce warp control divergence,
	block size should be a multiple of 2*warp size, 32 being the current value of warps in Fermi and Kepler GPU's*/

	int point1;
	int point2;
	int * child = new int[CHROMOSOME_SIZE];
	thrust::minstd_rand0 rng(seeds[threadIdx.x+blockDim.x*blockIdx.x]);

	thrust::uniform_int_distribution<int> dist1 = thrust::uniform_int_distribution<int>(0, CHROMOSOME_SIZE-1);
	point1 = dist1(rng);
	thrust::uniform_int_distribution<int> dist2 = thrust::uniform_int_distribution<int>(point1, CHROMOSOME_SIZE-1);
	point2 = dist2(rng);

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		child[i] = parent2[i];
	}
	__syncthreads();

	for(int j = point1; j <= point2; j++){
		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			if(child[i] == parent1[j]){
				int temp = child[j];
				child[j] = child[i];
				child[i] = temp;
			}
		}
	}

	__syncthreads();
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		parent1[i] = child[i];
	}

	distanceCalculation();
	delete child;
}

__device__ void geneticAlgorithm::distanceCalculation(){
	islandPopulationDistance[threadIdx.x] = distanceCalculation(islandPopulationChromosome[threadIdx.x]);
}

__device__ double geneticAlgorithm::distanceCalculation(int * chromosome){
	double distance = distanceBetweenTwoCities(chromosome[CHROMOSOME_SIZE-1], chromosome[0]);
	for(unsigned int i = 1; i < CHROMOSOME_SIZE; i++){
		unsigned int j  = i - 1;
		distance += distanceBetweenTwoCities(chromosome[i], chromosome[j]);
	}
	return distance;
}

__device__ double geneticAlgorithm::distanceBetweenTwoCities(int i, int j){
	return adjacencyMatrix[i*CHROMOSOME_SIZE+j];
}



#endif
