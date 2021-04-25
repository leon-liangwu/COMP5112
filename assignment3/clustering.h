#pragma once

#include <cmath>
#include <cstring>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda.h>
inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
};

#define GPUErrChk(ans) { GPUAssert((ans), __FILE__, __LINE__); }

void read_file(std::string file_path, int& num_vs, int& num_es, int*& nbr_offs, int*& nbrs);

void write_result_to_file(int nv, int num_cluster, int *res, std::string &path);

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result);
