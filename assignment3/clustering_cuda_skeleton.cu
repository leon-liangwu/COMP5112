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

int get_num_com_nbrs(int *nbrs, int left_start, int left_end, int right_start, int right_end) {
    int left_pos = left_start, right_pos = right_start, num_com_nbrs = 0;

    while (left_pos < left_end && right_pos < right_end) {
        if (nbrs[left_pos] == nbrs[right_pos]) {
            num_com_nbrs++;
            left_pos++;
            right_pos++;
        } else if (nbrs[left_pos] < nbrs[right_pos]) {
            left_pos++;
        } else {
            right_pos++;
        }
    }
    return num_com_nbrs;
}

// Device code
__global__ void find_pivots(int *nbr_offs, int *nbrs, bool* pivots, int* num_sim_nbrs, int* sim_nbrs, 
                        int *index_sims, int num_vs, float epsilon, int mu)
{
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	const int nthread = blockDim.x*gridDim.x;


    for(int i = tid; i < num_vs; i += nthread) {
		int left_start = nbr_offs[i];
        int left_end = nbr_offs[i + 1];
        int left_size = left_end - left_start;

        // loop over all neighbors of i
        for (int j = left_start; j < left_end; j++) {
            int nbr_id = nbrs[j];

            int right_start = nbr_offs[nbr_id];
            int right_end = nbr_offs[nbr_id + 1];
            int right_size = right_end - right_start;

            // compute the similarity
            int num_com_nbrs = 0;
            int left_pos = left_start, right_pos = right_start;
            while (left_pos < left_end && right_pos < right_end) {
                if (nbrs[left_pos] == nbrs[right_pos]) {
                    num_com_nbrs++;
                    left_pos++;
                    right_pos++;
                } else if (nbrs[left_pos] < nbrs[right_pos]) {
                    left_pos++;
                } else {
                    right_pos++;
                }
            }

            float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

            if (sim > epsilon) {
                sim_nbrs[index_sims[i]+num_sim_nbrs[i]] = nbr_id;
                num_sim_nbrs[i]++;
            }
        }
        if (num_sim_nbrs[i] > mu) pivots[i] = true;
        
	}
}

void expansion(int cur_id, int num_clusters, int *num_sim_nbrs, int **sim_nbrs,
               bool *visited, bool *pivots, int *cluster_result) {
    for (int i = 0; i < num_sim_nbrs[cur_id]; i++) {
        int nbr_id = sim_nbrs[cur_id][i];
        if ((pivots[nbr_id])&&(!visited[nbr_id])){
            visited[nbr_id] = true;
            cluster_result[nbr_id] = num_clusters;
            expansion(nbr_id, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots,
                        cluster_result);
        }
    }
}

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Fill in the cuda_scan function here
    bool *pivots = new bool[num_vs]();
    int *num_sim_nbrs = new int[num_vs]();
    int **sim_nbrs = new int*[num_vs];
    
    int *index_sims = new int[num_vs]();
    int *num_nbrs = new int[num_vs]();

    int index = 0;
    for(int i=0; i<num_vs; i++){
        int left_start = nbr_offs[i];
        int left_end = nbr_offs[i + 1];
        int left_size = left_end - left_start;
        sim_nbrs[i] = new int[left_size];
        num_nbrs[i] = left_size;
        index_sims[i] = index;
        index += left_size;
    }
    
    bool *d_pivots;
    int *d_nbr_offs, *d_nbrs, *d_num_sim_nbrs;
    int *d_sim_nbrs, *d_index_sims;

    // Allocate device memory
    cudaMalloc((void**)&d_nbr_offs, (num_vs+1) * sizeof(int));
    cudaMalloc((void**)&d_nbrs, (num_es+1) * sizeof(int));
    cudaMalloc((void**)&d_pivots, num_vs * sizeof(bool));
    cudaMalloc((void**)&d_num_sim_nbrs, num_vs * sizeof(int));
    cudaMalloc((void**)&d_sim_nbrs, index * sizeof(int));
    cudaMalloc((void**)&d_index_sims, num_vs * sizeof(int));

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_nbr_offs, nbr_offs, (num_vs+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbrs, nbrs, (num_es+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pivots, pivots, num_vs * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_sim_nbrs, num_sim_nbrs, num_vs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_sims, index_sims, num_vs * sizeof(int), cudaMemcpyHostToDevice);

    // stage 1:
    find_pivots<<<num_blocks_per_grid, num_threads_per_block>>>
        (d_nbr_offs, d_nbrs, d_pivots, d_num_sim_nbrs, d_sim_nbrs, d_index_sims, num_vs, epsilon, mu);

    cudaDeviceSynchronize();
    
    cudaMemcpy(nbr_offs, d_nbr_offs, (num_vs+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nbrs, d_nbrs, (num_es+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(pivots, d_pivots, num_vs * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_sim_nbrs, d_num_sim_nbrs, num_vs * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<num_vs; i++){
        if(num_sim_nbrs[i] > 0){
            cudaMemcpy(sim_nbrs[i], d_sim_nbrs+index_sims[i], num_nbrs[i] * sizeof(int), cudaMemcpyDeviceToHost);
        }    
    }


    // Stage 2:
    bool *visited = new bool[num_vs]();
    for (int i = 0; i < num_vs; i++) {
        if (!pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i, num_sim_nbrs, sim_nbrs, visited, pivots, cluster_result);

        num_clusters++;
    }

    // Free host memory
    for(int i=0; i<num_vs; i++){
        delete[] sim_nbrs[i];
    }
    delete[] sim_nbrs;
    delete[] index_sims;
    delete[] num_nbrs;
    delete[] pivots;
    delete[] num_sim_nbrs;
    delete[] visited;
    
    // Free device memory
    cudaFree(d_pivots);
    cudaFree(d_nbr_offs);
    cudaFree(d_nbrs);
    cudaFree(d_num_sim_nbrs);
    cudaFree(d_sim_nbrs);
    cudaFree(d_index_sims);
 
}
