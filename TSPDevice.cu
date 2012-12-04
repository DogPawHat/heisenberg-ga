#include "cuda.h"


class TSPDevice{
private:
	__device__ double** list;
	__device__ int size;
public:
	__device__ TSPDevice(double** list, int size){
		this.list = list;
		this.size = size;
	}

	__device__ __host__ static double GetDistance(double[] a, double[] b){
		double xd = a[0] - b[0];
		double yd = a[1] - b[1];
		return nint(sqrt( xd*xd + yd*yd));
	}
}