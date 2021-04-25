/* The Sequential Version
 * COMPILE: g++ -lstdc++ -std=c++11 clustering_impl.cpp main.cpp -o sequential
 * RUN:     ./sequential <path> <epsilon> <mu>
 */

#include <cassert>
#include <chrono>
#include <iostream>
#include "clustering.h"

using namespace std;

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

int main(int argc, char **argv) {
    assert(argc > 3 && "Usage: ./sequential <dataset_path> <epsilon> <mu>");

    std::string file_path = argv[1];
    float epsilon = std::stof(argv[2]);
    int mu = std::stoi(argv[3]);
    int num_vs, num_es, *nbr_offs = nullptr, *nbrs = nullptr;
    read_file(file_path, num_vs, num_es, nbr_offs, nbrs);

    int num_clusters = 0;
    int *cluster_result = new int[num_vs];
    std::fill(cluster_result, cluster_result + num_vs, -1);

    auto start_clock = chrono::high_resolution_clock::now();

    bool *pivots = new bool[num_vs]();
    int *num_sim_nbrs = new int[num_vs]();
    int **sim_nbrs = new int*[num_vs];

    // Stage 1:
    for (int i = 0; i < num_vs; i++) {
        int left_start = nbr_offs[i];
        int left_end = nbr_offs[i + 1];
        int left_size = left_end - left_start;

        sim_nbrs[i] = new int[left_size];
        // loop over all neighbors of i
        for (int j = left_start; j < left_end; j++) {
            int nbr_id = nbrs[j];

            int right_start = nbr_offs[nbr_id];
            int right_end = nbr_offs[nbr_id + 1];
            int right_size = right_end - right_start;

            // compute the similarity
            int num_com_nbrs = get_num_com_nbrs(nbrs, left_start, left_end, right_start, right_end);

            float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

            if (sim > epsilon) {
                sim_nbrs[i][num_sim_nbrs[i]] = nbr_id;
                num_sim_nbrs[i]++;
            }
        }
        if (num_sim_nbrs[i] > mu) pivots[i] = true;
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


    auto end_clock = chrono::high_resolution_clock::now();

    chrono::duration<double> diff = end_clock - start_clock;
    printf("Elapsed Time: %.9lf s\n", diff.count());
    printf("Number of clusters: %d\n", num_clusters);

    string result_path("results/sequential.txt");
    write_result_to_file(num_vs, num_clusters, cluster_result, result_path);

    delete[] pivots;
    delete[] num_sim_nbrs;
    delete[] sim_nbrs;
    delete[] visited;

    return 0;
    }
