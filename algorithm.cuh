#ifndef ALGORITHM_CUH
#define ALGORITHM_CUH

#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

class metaChromosome;
class geneticAlgorithm;


extern __shared__ int memoryPool[];
__shared__ int ** islandPopulationChromosome;
__shared__ double * islandPopulationDistance;
__shared__ int * warpCrossoverChance;
__shared__ int * warpMutationChance;
__shared__ int * rank;

class geneticAlgorithm{
public:
	const int BLOCK_SIZE;
	const int GRID_SIZE;
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
	bool * optimalLengthReached;



	__host__ geneticAlgorithm(const int GRID_SIZE, const int BLOCK_SIZE, const int GENERATIONS, const int OPTIMAL_LENGTH, const int CHROMOSOME_SIZE, const int CROSSOVER_CHANCE = 90, const int MUTATION_CHANCE = 25)
	: BLOCK_SIZE(BLOCK_SIZE),
	  GRID_SIZE(GRID_SIZE),
	  GENERATIONS(GENERATIONS),
	  OPTIMAL_LENGTH(OPTIMAL_LENGTH),
	  CHROMOSOME_SIZE(CHROMOSOME_SIZE),
	  POPULATION_SIZE(BLOCK_SIZE*GRID_SIZE),
	  ISLAND_POPULATION_SIZE(BLOCK_SIZE),
	  CROSSOVER_CHANCE(CROSSOVER_CHANCE),
	  MUTATION_CHANCE(MUTATION_CHANCE) ,
	  runGenerations(0)
	{
		optimalLengthReached = new bool[GRID_SIZE];
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
	__device__ void rankSelection();
	__device__ void roulletteSelection();

	__device__ void sort();
	__device__ void exchange(int *, int *);

	__device__ void crossover();
	__device__ void crossoverPMX(int*, int*);
	__device__ void crossoverERX(int *, int *);
	__device__ void crossoverGX(int *, int *);

	__device__ void mutation();
	__device__ void inversionMutation();
	__device__ void greedyMutation();


	__host__ __device__ double distanceBetweenTwoCities(int, int);
};

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

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
	delete tempResult;
}

__global__ void createRandomSeeds(geneticAlgorithm algorithm, long seed){

	thrust::minstd_rand0 rng(hash(seed*(threadIdx.x + blockIdx.x*blockDim.x)));

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	algorithm.seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


__global__ void runOneGeneration(geneticAlgorithm algorithm){
	algorithm.generation();
}

__device__ void geneticAlgorithm::generation(){
	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;


	warpCrossoverChance = (int*) memoryPool;
	warpMutationChance = (int*) &warpCrossoverChance[ISLAND_POPULATION_SIZE/32];
	islandPopulationChromosome = (int**) &warpMutationChance[ISLAND_POPULATION_SIZE/32];
	rank = (int*) &islandPopulationChromosome[ISLAND_POPULATION_SIZE];
	islandPopulationDistance = (double*) &rank[ISLAND_POPULATION_SIZE];

	thrust::minstd_rand rng(seeds[gridIndex]);
	thrust::uniform_int_distribution<short> dist(1, 100);

	if(threadIdx.x % 32 == 0){
		warpCrossoverChance[threadIdx.x/32] = dist(rng);
		warpMutationChance[threadIdx.x/32] = dist(rng);
	}
	__syncthreads();

	
	islandPopulationChromosome[threadIdx.x] = &populationChromosome[gridIndex*CHROMOSOME_SIZE];


	distanceCalculation();
	__syncthreads();

	selection();
	__syncthreads();

	if(warpCrossoverChance[threadIdx.x/32] < CROSSOVER_CHANCE){
		crossover();
	}
	__syncthreads();

	if(warpMutationChance[threadIdx.x/32] < MUTATION_CHANCE){
		mutation();
	}
	__syncthreads();

	distanceCalculation();
	createNewSeed(seeds[gridIndex]);
	__syncthreads();

	sort();
	__syncthreads();
	
	distanceCalculation();
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
	if(threadIdx.x >= (ISLAND_POPULATION_SIZE/divisor)*3){
		int migrationBlockOffset = 1/*threadIdx.x/(ISLAND_POPULATION_SIZE/divisor)*/;
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
	thrust::minstd_rand rng(hash(seed));

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


/*Selection Functions*/

__device__ void geneticAlgorithm::selection(){
	rankSelection();
}

__device__ void geneticAlgorithm::tournamentSelection(){
	int N = 1;
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

__device__ void geneticAlgorithm::rankSelection(){
	sort();
	__syncthreads();
	int * selection = new int[CHROMOSOME_SIZE];


	rank[threadIdx.x] = ISLAND_POPULATION_SIZE-threadIdx.x;
	__syncthreads();
	for(int stride = 1; stride < ISLAND_POPULATION_SIZE; stride *= 2){
		int index = threadIdx.x*2*stride;
		if(index+stride < ISLAND_POPULATION_SIZE){
			rank[index] += rank[index+stride];
		}
		__syncthreads();
	}

	for(int stride = ISLAND_POPULATION_SIZE/4; stride > 0; stride /= 2){
		__syncthreads();
		int index = (threadIdx.x+1)*2*stride;
		if (index < ISLAND_POPULATION_SIZE && index-stride > 0)
		{
			rank[index-stride] += rank[index];
		}
	}
	__syncthreads();

	thrust::minstd_rand rng(seeds[threadIdx.x+blockIdx.x*blockDim.x]);
	thrust::uniform_int_distribution<int> dist(rank[blockDim.x-1], rank[0]-1);

	int roulletteBall = dist(rng);

	for(int i = BLOCK_SIZE-1; i >= 0; i--){
		if(roulletteBall > rank[i] && roulletteBall <= rank[i-1]){
			for(int j = 0; j < CHROMOSOME_SIZE; j++){
				selection[j] = islandPopulationChromosome[i][j];
			}
			break;
		}
	}

	__syncthreads();
	for(int j = 0; j < CHROMOSOME_SIZE; j++){
		islandPopulationChromosome[threadIdx.x][j] = selection[j];
	}

	delete selection;
	__syncthreads();

	distanceCalculation();
	__syncthreads();
}


/* Sorting Algorithms */

__device__ void geneticAlgorithm::sort(){

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
	greedyMutation();
}

__device__ void geneticAlgorithm::inversionMutation(){
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

__device__ void geneticAlgorithm::greedyMutation(){
	int * mutant = new int[CHROMOSOME_SIZE];
	thrust::minstd_rand0 rng(seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	int swapPoint1;
	int swapPoint2;

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		mutant[i] = islandPopulationChromosome[threadIdx.x][i];
	}

	thrust::uniform_int_distribution<int> dist1(0, CHROMOSOME_SIZE-1);
	swapPoint1 = dist1(rng);
	thrust::uniform_int_distribution<int> dist2(swapPoint1, CHROMOSOME_SIZE-1);
	swapPoint2 = dist2(rng);

	int * currentCity;
	int * nextCity;

	for(int i = swapPoint1; i < swapPoint2; i++){
		currentCity = &mutant[i];
		nextCity = &mutant[i+1];
		for(int j = i+2; j < swapPoint2; j++){
			if(distanceBetweenTwoCities(*currentCity, mutant[j]) > distanceBetweenTwoCities(*currentCity, *nextCity)){
				nextCity = &mutant[j];
			}
		}
		int temp = mutant[i];
		mutant[i] = *nextCity;
		*nextCity = temp;
	}

	__syncthreads();

	if(distanceCalculation(mutant) < distanceCalculation(islandPopulationChromosome[threadIdx.x])){
		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			islandPopulationChromosome[threadIdx.x][i] = mutant[i];
		}
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

__device__ void geneticAlgorithm::crossoverERX(int * parent1, int * parent2){

	int ** edgeList = new int*[CHROMOSOME_SIZE];
	int * child = new int[CHROMOSOME_SIZE];

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		edgeList[i] = new int[4];
		for(int j = 0; j < CHROMOSOME_SIZE; j++){
			for(int k = 0; k < CHROMOSOME_SIZE; k++){
				if(parent1[j] == i && parent2[k] == i){
					int xa, xb, ya, yb;

					if(j == 0){
						xa = parent1[CHROMOSOME_SIZE - 1];
						xb = parent1[j+1];
					}else if(j == (CHROMOSOME_SIZE-1)){
						xa = parent1[j - 1];
						xb = parent1[0];
					}else{
						xa = parent1[j - 1];
						xb = parent1[j+1];
					}

					if(k == 0){
						ya = parent2[CHROMOSOME_SIZE - 1];
						yb = parent2[k+1];
					}else if(k == (CHROMOSOME_SIZE-1)){
						ya = parent2[k-1];
						yb = parent2[0];
					}else{
						ya = parent2[k-1];
						yb = parent2[k+1];
					}

					edgeList[i][0] = xa;
					edgeList[i][1] = xb;
					if(xa != ya || xb != ya){
						edgeList[i][2] = ya;
					}else{
						edgeList[i][2] = CHROMOSOME_SIZE;
					}

					if(xa != yb || xb != yb){
						edgeList[i][3] = yb;
					}
					else
					{
						edgeList[i][3] = CHROMOSOME_SIZE;
					}
					break;
				}
			}
		}
	}

	int currentNode = parent2[0];


	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		if(currentNode < CHROMOSOME_SIZE && currentNode >= 0){
			child[i] = currentNode;

			for(int j = 0; j < CHROMOSOME_SIZE; j++){
				for(int k = 0; k < 4; k++){
					if(currentNode==edgeList[j][k]){
						edgeList[j][k] = CHROMOSOME_SIZE;
						break;
					}
				}
			}

			bool nonEmpty = false;
			for(int j = 0; j < 4; j++){
				if(edgeList[currentNode][j] < CHROMOSOME_SIZE){
					nonEmpty = true;
					break;
				}
			}


			if(nonEmpty){
				short currentListSize = 4;
				short listSize= 0;
				for(int j = 0; j < 4; j++){
					if(edgeList[currentNode][j] < CHROMOSOME_SIZE){
						listSize = 0;
						for(int k = 0; k < 4; k++){
							if(edgeList[edgeList[currentNode][j]][k] != CHROMOSOME_SIZE){
								listSize++;
							}
						}

						if(listSize <= currentListSize){
							currentNode = edgeList[currentNode][j];
							currentListSize = listSize;
						}
					}
				}
			}
			else if(i<CHROMOSOME_SIZE-1){
				int j = currentNode;
				int k = 4;
				do{
					if(j+1<CHROMOSOME_SIZE){
						j = j+1;
					}else{
						j = 0;
					}

					if(k+1<4){
						k = k+1;
					}else{
						k = 0;
					}


					currentNode= edgeList[i][j];
				}while(currentNode == CHROMOSOME_SIZE);
			}else{
				break;
			}
		}
	}

	__syncthreads();
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulationChromosome[threadIdx.x][i] = child[i];
	}
	__syncthreads();


//	distanceCalculation();
//	__syncthreads();
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
	for(int i = 1; i < CHROMOSOME_SIZE; i++){
		distance += distanceBetweenTwoCities(chromosome[i], chromosome[i-1]);
	}
	return distance;
}

__device__ double geneticAlgorithm::distanceBetweenTwoCities(int i, int j){
	return adjacencyMatrix[i*CHROMOSOME_SIZE+j];
}

#endif
