#include <cuda.h>
#include <math_functions.h>

__device__ static double GetDistance(double* a, double* b){
	double xd = a[0] - b[0];
	double yd = a[1] - b[1];
	return ceilf(sqrtf( xd*xd + yd*yd));
}