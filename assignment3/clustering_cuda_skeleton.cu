/* 
 * Name: WU Liang 
 * USTid: 20636843 
 * Email: lwuat@connect.ust.hk
 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp main.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>
 */

#include <iostream>
#include "clustering.h"

// Define variables or functions here

// Device code
__global__ void find_pivots(bool* pivots, int* num_sim_nbrs, int* sim_nbrs, int num_vs)
{
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	const int nthread = blockDim.x*gridDim.x;

    for(int i = tid; i < numElement; i += nthread) {
		pivots = true;
	}
}


void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Fill in the cuda_scan function here
    bool *pivots = new bool[num_vs]();
    int *num_sim_nbrs = new int[num_vs]();
    int **sim_nbrs = new int*[num_vs];

    # stage 1:

    
}
