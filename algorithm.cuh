#ifndef ALGORITHM_CUH
#define ALGORITHM_CUH

#include <cuda.h>
#include <math.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/uniform_int_distribution.h>


class metaChromosome;
class geneticAlgorithm;

extern __shared__ int sharedMemoryPool[];
__shared__ int * islandPopulationChromosome;
__shared__ double * islandPopulationDistance;

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
	int * source;
	int * seeds;
	double * adjacencyMatrix;
	int * populationChromosome;
	double * populationDistance;


	__host__ __device__ geneticAlgorithm(const int BLOCK_SIZE, const int GRID_SIZE, const int GENERATIONS, const int CHROMOSOME_SIZE, const int CROSSOVER_CHANCE = 75, const int MUTATION_CHANCE = 10)
		: BLOCK_SIZE(BLOCK_SIZE),
		GRID_SIZE(GRID_SIZE),
		GENERATIONS(GENERATIONS),
		CHROMOSOME_SIZE(CHROMOSOME_SIZE),
		POPULATION_SIZE(BLOCK_SIZE*GRID_SIZE),
		ISLAND_POPULATION_SIZE(BLOCK_SIZE),
		CROSSOVER_CHANCE(CROSSOVER_CHANCE),
		MUTATION_CHANCE(MUTATION_CHANCE) 
	{}

	__device__ void generation();
	__device__ void distanceCalculation();

private:

	__device__ void migration(int);

	__device__ void createNewSeed(long);

	__device__ void selection();
	__device__ void rouletteSelection();
	__device__ float randomRouletteBall();
	__device__ void fitnessEvauation(float[]);
	__device__ void tournamentSelection();

	__device__ void sort();
	__device__ void exchange(int *, int *);

	__device__ void crossover();
	__device__ void crossoverOX(int*, int*);
//	__device__ void crossoverERO(metaChromosome*, metaChromosome*);

	__device__ void mutation();

	__host__ __device__ double distanceCalculation(int*);
	__host__ __device__ double distanceBetweenTwoCities(int, int);
};

__global__ void createRandomPermutation(geneticAlgorithm* algorithm){
	int * tempResult = (int*) &sharedMemoryPool[threadIdx.x*algorithm->CHROMOSOME_SIZE];
	int temp;
	int rand;
	int * chromosome = &(algorithm->populationChromosome[(threadIdx.x+blockIdx.x*blockDim.x)*algorithm->CHROMOSOME_SIZE]);

	thrust::minstd_rand0 rng(algorithm->seeds[threadIdx.x+blockIdx.x*blockDim.x]);

	for(int i = 0; i < algorithm->CHROMOSOME_SIZE; i++){
		tempResult[i] = algorithm->source[i];
	}

	for(int i = algorithm->CHROMOSOME_SIZE-1; i >= 0; i--){
		thrust::uniform_int_distribution<int> dist(0,i);
		rand = dist(rng);
		temp = tempResult[rand];
		tempResult[rand] = tempResult[i];
		tempResult[i] = temp;
	}
	__syncthreads();

	for(int i = 0; i < algorithm->CHROMOSOME_SIZE; i++){
		chromosome[i] = tempResult[i];
	}
	algorithm->distanceCalculation();
}

__global__ void createRandomSeeds(geneticAlgorithm* algorithm, long seed){
	thrust::minstd_rand0 rng(seed*(threadIdx.x + blockIdx.x*blockDim.x));

	thrust::uniform_int_distribution<int> dist(0,RAND_MAX);
	algorithm->seeds[threadIdx.x + blockDim.x*blockIdx.x]=dist(rng);
}


__global__ void runOneGeneration(geneticAlgorithm* algorithm){
	algorithm->generation();
}

__device__ void geneticAlgorithm::generation(){
	islandPopulationChromosome = (int*) &sharedMemoryPool[0];
	islandPopulationDistance = (double*) &sharedMemoryPool[CHROMOSOME_SIZE*POPULATION_SIZE];

	int gridIndex = threadIdx.x + blockDim.x*blockIdx.x;
	islandPopulationDistance[threadIdx.x] = populationDistance[gridIndex];
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulationChromosome[threadIdx.x*CHROMOSOME_SIZE+i] = populationChromosome[gridIndex*CHROMOSOME_SIZE+i];
	}
	__syncthreads();

	thrust::minstd_rand rng(seeds[gridIndex]);
	thrust::uniform_int_distribution<short> dist(1, 100);

	distanceCalculation();
	__syncthreads();

	selection();
	__syncthreads();
/*
	if(dist(rng) < CROSSOVER_CHANCE){
		crossover();
		__syncthreads();
	}

	if(dist(rng) < MUTATION_CHANCE){
		mutation();
		__syncthreads();
	}
*/
	createNewSeed(seeds[gridIndex]);
	__syncthreads();

//	sort();
//	__syncthreads();

	distanceCalculation();
	__syncthreads();

	migration(gridIndex);
	__syncthreads();
}




/* Migration Functions */

__device__ void geneticAlgorithm::migration(int gridIndex){
	if(threadIdx.x < BLOCK_SIZE/2){
		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			populationChromosome[gridIndex*CHROMOSOME_SIZE+i] = islandPopulationChromosome[threadIdx.x*CHROMOSOME_SIZE+i];
		}
		populationDistance[gridIndex] = islandPopulationDistance[threadIdx.x];
	}else if(blockIdx.x < GRID_SIZE - 1){
		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			populationChromosome[(gridIndex+(ISLAND_POPULATION_SIZE))*CHROMOSOME_SIZE+i] = islandPopulationChromosome[(threadIdx.x-(ISLAND_POPULATION_SIZE/2))*CHROMOSOME_SIZE+i];
		}
		populationDistance[gridIndex+(ISLAND_POPULATION_SIZE)] = islandPopulationDistance[threadIdx.x-(ISLAND_POPULATION_SIZE/2)];
	}else{
		for(int i = 0; i < CHROMOSOME_SIZE; i++){
			populationChromosome[threadIdx.x*CHROMOSOME_SIZE+i] = islandPopulationChromosome[(threadIdx.x-(ISLAND_POPULATION_SIZE/2))*CHROMOSOME_SIZE+i];
		}
		populationDistance[threadIdx.x] = islandPopulationDistance[threadIdx.x-(ISLAND_POPULATION_SIZE/2)];
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



/*__device__ void geneticAlgorithm::rouletteSelection(){ //Dodge as fuck
	metaChromosome selectedIndividual;
	__shared__ float fitnessValues[ISLAND_POPULATION_SIZE];
	float rouletteBall;

	fitnessEvauation(fitnessValues);
	__syncthreads();

	rouletteBall = randomRouletteBall();

	for(short i = 1; i < ISLAND_POPULATION_SIZE; i++){
		if(rouletteBall < fitnessValues[i]){
			selectedIndividual = islandPopulation[i];
			break;
		}
	}

	islandPopulation[threadIdx.x] = selectedIndividual;
	__syncthreads();
}*/

/* __device__ float geneticAlgorithm::randomRouletteBall(){
	thrust::minstd_rand0 rng(device.seeds[threadIdx.x + blockDim.x*blockIdx.x]);
	thrust::uniform_real_distribution<float> dist(0, 1);
	float result = dist(rng);
	return result;
}*/

/*__device__ void geneticAlgorithm::fitnessEvauation(float fitnessValues[]){
	fitnessValues[threadIdx.x] = islandPopulation[ISLAND_POPULATION_SIZE-1].distance - islandPopulation[threadIdx.x].distance;
	__syncthreads();

	for(int stride = 1; stride < ISLAND_POPULATION_SIZE; stride *= 2){
		if(threadIdx.x + stride < ISLAND_POPULATION_SIZE){
			fitnessValues[threadIdx.x] += fitnessValues[threadIdx.x+stride];
			__syncthreads();
		}
	}

	fitnessValues[threadIdx.x] = fitnessValues[threadIdx.x]/fitnessValues[0];
}*/


__device__ void geneticAlgorithm::tournamentSelection(){
	int N = 5;
	int tournamentChampion;
	int tournamentChallenger;

	thrust::minstd_rand rng(seeds[threadIdx.x + blockIdx.x*blockDim.x]);
	thrust::uniform_int_distribution<short> dist(0, CHROMOSOME_SIZE-1);
	
	tournamentChampion = threadIdx.x;

	for(int i = 0; i < N; i++){
		tournamentChallenger = dist(rng);
		if(islandPopulationDistance[tournamentChampion] > islandPopulationDistance[tournamentChallenger]){
			tournamentChampion = tournamentChallenger;
		}
	}
	__syncthreads();

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		islandPopulationChromosome[threadIdx.x*CHROMOSOME_SIZE+i] = islandPopulationChromosome[tournamentChampion*CHROMOSOME_SIZE+i];
	}
	islandPopulationDistance[threadIdx.x] = islandPopulationDistance[tournamentChampion];
}

/* Sorting Algorithms */

__device__ void geneticAlgorithm::sort(){
	for (int k = 2; k <= ISLAND_POPULATION_SIZE; k <<= 1){
		__syncthreads();
		for (int j=k>>1; j>0; j=j>>1){
			__syncthreads();
			unsigned int i = threadIdx.x; 
			int ixj = i^j;
		
			if ((ixj)>i){
				if ((i&k)==0 && islandPopulationDistance[i]>islandPopulationDistance[ixj]){
					exchange(&islandPopulationChromosome[i*CHROMOSOME_SIZE], &islandPopulationChromosome[ixj*CHROMOSOME_SIZE]);
					int temp = islandPopulationDistance[i];
					islandPopulationDistance[i] = islandPopulationDistance[ixj];
					islandPopulationDistance[ixj] = temp;
					__syncthreads();
				}
				if ((i&k)!=0 && islandPopulationDistance[i]<islandPopulationDistance[ixj]){
					exchange(&islandPopulationChromosome[i*CHROMOSOME_SIZE], &islandPopulationChromosome[ixj*CHROMOSOME_SIZE]);
					int temp = islandPopulationDistance[i];
					islandPopulationDistance[i] = islandPopulationDistance[ixj];
					islandPopulationDistance[ixj] = temp;
					__syncthreads();
				}
			}
		}
	}
}

__device__ void geneticAlgorithm::exchange(int * chromosome1, int * chromosome2){
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		int temp = chromosome1[i];
		chromosome1[i] = chromosome2[i];
		chromosome2[i] = temp;
	}
}


/*Genetic Operators*/

__device__ void geneticAlgorithm::mutation(){
	int * mutant = &islandPopulationChromosome[threadIdx.x*CHROMOSOME_SIZE];
	thrust::minstd_rand0 rng(seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	thrust::uniform_int_distribution<short> dist1(0, 10);
	thrust::uniform_int_distribution<short> dist2(0, CHROMOSOME_SIZE-1);
	int numOfSwaps = dist1(rng);
	int swapPoint1;
	int swapPoint2;
	int temp;

	for(int i = 0; i < numOfSwaps; i++){
		swapPoint1 = dist2(rng);
		swapPoint2 = dist2(rng);
		temp = mutant[swapPoint1];
		mutant[swapPoint1] = mutant[swapPoint2];
		mutant[swapPoint2] = temp;
	}

	distanceCalculation();
}

__device__ void geneticAlgorithm::crossover(){
	int * parent1;
	int * parent2;
	
	if(threadIdx.x < (BLOCK_SIZE/2)){
		parent1 = &islandPopulationChromosome[threadIdx.x];
		parent2 = &islandPopulationChromosome[threadIdx.x+(BLOCK_SIZE/2)];
	}else{
		parent1 = &islandPopulationChromosome[threadIdx.x];
		parent2 = &islandPopulationChromosome[threadIdx.x-(BLOCK_SIZE/2)];
	}

	crossoverOX(parent1, parent2);
}

__device__ void geneticAlgorithm::crossoverOX(int * parent1, int * parent2){
	/*We need two different paths here beause each thread needs two parents to generate a single offspring.
	The first half of the block will take one parent from the first half of islandPopulation, while the second parent
	will come from the second half. This is reversed for the second half of the block. To reduce warp control divergence,
	block size shoud be a multiple of 2*warp size, 32 being the current value of warps in Fermi and Kepler GPU's*/
	
	short point1;
	short point2;
	thrust::minstd_rand0 rng(seeds[threadIdx.x+blockDim.x*blockIdx.x]);
	thrust::uniform_int_distribution<short> dist1;
	thrust::uniform_int_distribution<short> dist2;

	dist1 = thrust::uniform_int_distribution<short>(0, CHROMOSOME_SIZE-1);
	point1 = dist1(rng);
	dist2 = thrust::uniform_int_distribution<short>(point1, CHROMOSOME_SIZE-1);
	point2 = dist2(rng);

	int position = 0;
	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		if(!(position < point1 || position > point2)){
			position = point2 + 1;
		}
		if(!(position < CHROMOSOME_SIZE)){
			break;
		}


		bool nonDuplicate = true;
		for(short j = point1; j <= point2; j++){
			if(parent1[j] == parent2[i]){
				nonDuplicate = false;
				break;
			}
		}
		if(nonDuplicate == true){
			parent1[position] = parent2[i];
			position++;
		}
	}

	distanceCalculation();
}

/*__device__ void geneticAlgorithm::crossoverERO(metaChromosome * parent1, metaChromosome * parent2){
	thrust::minstd_rand rng(device.seeds[threadIdx.x]);
	thrust::uniform_int_distribution<short> dist(0, (CHROMOSOME_SIZE-1));
	unsigned short unionAjacency[CHROMOSOME_SIZE][4];
	metaChromosome child;
	unsigned short currentAvailable[CHROMOSOME_SIZE];
	unsigned short currentNode= dist(rng);

	for(unsigned short i = 0; i < CHROMOSOME_SIZE; i++){
		currentAvailable[i] = i;
	}

	currentAvailable[currentNode]= CHROMOSOME_SIZE + 1;

	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		for(int j = 0; j < CHROMOSOME_SIZE; j++){
			for(int k = 0; k < CHROMOSOME_SIZE; k++){
				if(parent1->chromosome[j] == i && parent2->chromosome[k] == i){
					unsigned short xa, xb, ya, yb;

					switch(j){
					case 0:
						xa = parent1->chromosome[CHROMOSOME_SIZE - 1];
						xb = parent1->chromosome[j+1];
						break;
					case CHROMOSOME_SIZE-1:
						xa = parent1->chromosome[j - 1];
						xb = parent1->chromosome[0];
						break;
					default:
						xa = parent1->chromosome[j - 1];
						xb = parent1->chromosome[j+1];
						break;
					}

					switch(k){
					case 0:
						ya = parent2->chromosome[CHROMOSOME_SIZE - 1];
						yb = parent2->chromosome[k+1];
						break;
					case CHROMOSOME_SIZE-1:
						ya = parent2->chromosome[k-1];
						yb = parent2->chromosome[0];
						break;
					default:
						ya = parent2->chromosome[k-1];
						yb = parent2->chromosome[k+1];
						break;
					}
					if(xa <= CHROMOSOME_SIZE && xb <= CHROMOSOME_SIZE && ya <= CHROMOSOME_SIZE && ya <= CHROMOSOME_SIZE){
						unionAjacency[i][0] = xa;
						unionAjacency[i][1] = xb;
						if(xa != ya || xb != ya){
							unionAjacency[i][2] = ya;
						}else{
							unionAjacency[i][2] = CHROMOSOME_SIZE+1;
						}

						if(xa != yb || xb != yb){
							unionAjacency[i][3] = yb;
						}
						else
						{
							unionAjacency[i][3] = CHROMOSOME_SIZE+1;
						}
						break;
					}
				}
			}
		}
	}


	for(int i = 0; i < CHROMOSOME_SIZE; i++){
		if(currentNode < CHROMOSOME_SIZE && currentNode >= 0){
			child.chromosome[i] = currentNode;
			currentAvailable[currentNode]= CHROMOSOME_SIZE + 1;

			for(int j = 0; j < CHROMOSOME_SIZE; j++){
				for(int k = 0; k < 4; k++){
					if(currentNode==unionAjacency[j][k]){
						unionAjacency[j][k] = CHROMOSOME_SIZE+1;
						break;
					}
				}
			}

			bool nonEmpty = false;
			for(int j = 0; j < 4; j++){
				if(unionAjacency[currentNode][j] < CHROMOSOME_SIZE){
					nonEmpty = true;
					break;
				}
			}


			if(nonEmpty == true){
				short currentListSize = 4;
				short listSize= 0;
				for(int j = 0; j < 4; j++){
					if(unionAjacency[currentNode][j] < CHROMOSOME_SIZE){
						listSize = 0;
						for(int k = 0; k < 4; k++){
							if(unionAjacency[unionAjacency[currentNode][j]][k] != CHROMOSOME_SIZE+1){
								listSize++;
							}
						}

						if(listSize <= currentListSize){
							currentNode = unionAjacency[currentNode][j];
							currentListSize = listSize;
						}
					}
				}
			}
			else{
				do{
					currentNode= dist(rng);
				}while(currentAvailable[currentNode]== CHROMOSOME_SIZE + 1);
			}
		}
	}

	child.distanceCalculation(device.TSPGraph);
	islandPopulation[threadIdx.x] = child;
}*/

__host__ __device__ double geneticAlgorithm::distanceCalculation(int * chromosome){
	double distance = distanceBetweenTwoCities(chromosome[CHROMOSOME_SIZE-1], chromosome[0]);
	for(unsigned int i = 1; i < CHROMOSOME_SIZE; i++){
		unsigned int j  = i - 1;
		distance += distanceBetweenTwoCities(chromosome[i], chromosome[j]);
	}
	return distance;
}

__host__ __device__ double geneticAlgorithm::distanceBetweenTwoCities(int i, int j){
	return adjacencyMatrix[i*CHROMOSOME_SIZE+j];
}

__device__ void geneticAlgorithm::distanceCalculation(){
	islandPopulationDistance[threadIdx.x] = distanceCalculation(&islandPopulationChromosome[threadIdx.x*CHROMOSOME_SIZE]);
}


#endif