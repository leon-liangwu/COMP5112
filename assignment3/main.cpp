/* Keep this file unchanged
 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>
 */

#include <cassert>
#include <chrono>
#include <iostream>
#include "clustering.h"

int main(int argc, char **argv) {
    assert(argc > 5 && "Usage: ./cuda <dataset_path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>");

    std::string file_path = argv[1];
    float epsilon = std::stof(argv[2]);
    int mu = std::stoi(argv[3]);
    int num_blocks_per_grid = std::stoi(argv[4]);
    int num_threads_per_block = std::stoi(argv[5]);

    int num_vs, num_es, *nbr_offs = nullptr, *nbrs = nullptr;
    read_file(file_path, num_vs, num_es, nbr_offs, nbrs);

    int num_clusters = 0;
    int *cluster_result = new int[num_vs];
    std::fill(cluster_result, cluster_result + num_vs, -1);

    cudaDeviceReset();
    cudaEvent_t cuda_start, cuda_end;

    float kernel_time;
    auto start_clock = std::chrono::high_resolution_clock::now();
    
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    cudaEventRecord(cuda_start);

    cuda_scan(
        num_vs, num_es, nbr_offs, nbrs,
        epsilon, mu, num_blocks_per_grid, num_threads_per_block,
        num_clusters, cluster_result
    );
    
    cudaEventRecord(cuda_end);

    cudaEventSynchronize(cuda_start);
    cudaEventSynchronize(cuda_end);

    cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);
    GPUErrChk(cudaDeviceSynchronize());

    auto end_clock = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end_clock - start_clock;
    
    printf("Elapsed Time: %.9lf s\n", diff.count());
    fprintf(stderr, "Driver Time: %.9lf s\n", kernel_time / pow(10, 3));
    printf("Number of clusters: %d\n", num_clusters);

    std::string result_path("results/cuda.txt");
    write_result_to_file(num_vs, num_clusters, cluster_result, result_path);

    delete[] cluster_result;
    delete[] nbr_offs;
    delete[] nbrs;

    return 0;
}
