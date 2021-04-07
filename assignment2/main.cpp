/* Keep this file unchanged
 * COMPILE: g++ -lstdc++ -std=c++11 -lpthread clustering_pthread_skeleton.cpp main.cpp -o pthread
 *                  ps, in some platform, the command "-lpthread" should be replaced by "-pthread"
 * RUN:     ./pthread <path> <epsilon> <mu> <num_threads>
 */

#include <cassert>
#include <chrono>
#include <iostream>

using namespace std;

int *scan(float epsilon, int mu, int num_threads, int num_vs, int num_es, int *nbr_offs, int *nbrs);
void write_result_to_file(int nv, int num_cluster, int *res, string &path);
void read_file(string file_path, int& num_vs, int& num_es, int*& nbr_offs, int*& nbrs);


int main(int argc, char **argv) {

    assert(argc > 4 && "Usage: ./reference <dataset_path> <epsilon> <mu> <num_threads>");

    std::string file_path = argv[1];
    float epsilon = std::stof(argv[2]);
    int mu = std::stoi(argv[3]);
    int num_threads = std::atoi(argv[4]);

    int num_vs, num_es, *nbr_offs = nullptr, *nbrs = nullptr;
    read_file(file_path, num_vs, num_es, nbr_offs, nbrs);

    auto start_clock = chrono::high_resolution_clock::now();

    int *cluster_result = scan(epsilon, mu, num_threads, num_vs, num_es, nbr_offs, nbrs);

    int num_clusters = 0;
    for (auto i = 0; i< num_vs; i++){
        if (cluster_result[i] == i)
            num_clusters ++;
    }

    auto end_clock = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_clock - start_clock;
    printf("Elapsed Time: %.9lf s\n", diff.count());
    printf("Number of clusters: %d\n", num_clusters);

    string result_path("results/parallel.txt");
    write_result_to_file(num_vs, num_clusters, cluster_result, result_path);

    return 0;
}