#ifndef ALGORITHMKERNEL_H
#define ALGORITHMKERNEL_H

__global__ void runGeneticAlgorithm(float * poplulation, float * tspGraph, int popLength, int graphSize);

#endif