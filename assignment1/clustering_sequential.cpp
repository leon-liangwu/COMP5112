#include "clustering.h"

#include <cassert>
#include <chrono>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  int *nbr_offs = nullptr, *nbrs = nullptr;
  int *nbr_offs_local = nullptr, *nbrs_local = nullptr;

  GraphMetaInfo *info = nullptr;

  if (argc != 3) {
    std::cerr << "usage: ./clustering_sequential data_path result_path"
              << std::endl;

    return -1;
  }
  std::string dir(argv[1]);
  std::string result_path(argv[2]);

  int num_graphs = read_files(dir, info, nbr_offs, nbrs);

  auto start_clock = chrono::high_resolution_clock::now();
  nbr_offs_local = nbr_offs;
  nbrs_local = nbrs;

  // cluster index for each vertices
  int *clustering_results[num_graphs];
  // number of clusters for each graph
  int num_cluster_total[num_graphs];

  for (size_t i = 0; i < num_graphs; i++) {
    GraphMetaInfo info_local = info[i];
    clustering_results[i] =
        (int *)calloc(info_local.num_vertices, sizeof(int));

    int num_cluster_local = clustering(info_local, nbr_offs_local, nbrs_local,
                                       clustering_results[i]);
    num_cluster_total[i] = num_cluster_local;

    printf("num cluster in graph %d : %d\n", i, num_cluster_local);

    nbr_offs_local += (info_local.num_vertices + 1);
    nbrs_local += (info_local.num_edges + 1);
  }
  auto end_clock = chrono::high_resolution_clock::now();

  fprintf(stderr, "Elapsed Time: %.9lf ms\n",
          chrono::duration_cast<chrono::nanoseconds>(end_clock - start_clock)
                  .count() /
              pow(10, 6));

  // write results to file
  for (int i = 0; i < num_graphs; i++) {
    GraphMetaInfo info_local = info[i];
    write_result_to_file(info_local, i, num_cluster_total[i],
                         clustering_results[i], result_path);
  }

  return 0;
}
