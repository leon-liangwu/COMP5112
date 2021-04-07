/*
 WU Liang 
 20636843 
 lwuat@connect.ust.hk
*/
#include "clustering.h"

#include "mpi.h"

#include <cassert>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm;
  int num_process; // number of processors
  int my_rank;     // my global rank

  comm = MPI_COMM_WORLD;

  MPI_Comm_size(comm, &num_process);
  MPI_Comm_rank(comm, &my_rank);

  if (argc != 3) {
    std::cerr << "usage: ./clustering_sequential data_path result_path"
              << std::endl;

    return -1;
  }
  std::string dir(argv[1]);
  std::string result_path(argv[2]);

  int num_graphs;
  int *clustering_results = nullptr;
  int *num_cluster_total = nullptr;

  int *nbr_offs = nullptr, *nbrs = nullptr;
  int *nbr_offs_local = nullptr, *nbrs_local = nullptr;

  GraphMetaInfo *info = nullptr;

  // read graph info from files
  if (my_rank == 0) {
    num_graphs = read_files(dir, info, nbr_offs, nbrs);
  }
  auto start_clock = chrono::high_resolution_clock::now();

  // ADD THE CODE HERE

  if (my_rank == 0) {
    int local_n = num_graphs / num_process;

    int total_vertices = 0;
    int* info_edges = (int *)calloc(num_graphs, sizeof(int));
    int* info_vertices = (int *)calloc(num_graphs, sizeof(int));

    for(size_t i=0; i<num_graphs; i++) {
      info_edges[i] = info[i].num_edges;
      info_vertices[i] = info[i].num_vertices;
      total_vertices += info[i].num_vertices;
    }

    clustering_results = (int *)calloc(total_vertices, sizeof(int));
    num_cluster_total = (int *)calloc(num_graphs, sizeof(int));

    int offs_index = 0;
    int nbrs_index = 0;
    int* num_vertices = (int *)calloc(num_process, sizeof(int));
    for(size_t i=0; i<num_process; i++) {
      num_vertices[i] = 0;
      int num_offs = 0;
      int num_nbrs = 0;
      int end_graph = (i+1) * local_n;
      int start_graph = i * local_n;
      for(size_t j=start_graph; j<end_graph; j++) {
        num_offs += info_vertices[j] + 1;
        num_nbrs += info_edges[j] + 1;
        num_vertices[i] += info_vertices[j];
      }

      if( i != 0) {
        MPI_Send(&local_n, 1, MPI_INT, i, 0, comm);
        MPI_Send(info_edges+start_graph, local_n, MPI_INT, i, 0, comm);
        MPI_Send(info_vertices+start_graph, local_n, MPI_INT, i, 0, comm);
        MPI_Send(nbr_offs+offs_index, num_offs, MPI_INT, i, 0, comm);
        MPI_Send(nbrs+nbrs_index, num_nbrs, MPI_INT, i, 0, comm);
      }

      offs_index += num_offs;
      nbrs_index += num_nbrs;
 
    }

    int* nbr_offs_local = nbr_offs;
    int* nbrs_local = nbrs;
    int* clustering_results_ptr = clustering_results;
    for(size_t i=0; i<local_n; i++) {
        GraphMetaInfo info_local = info[i];
        num_cluster_total[i] = clustering(info_local, nbr_offs_local, nbrs_local,
                                        clustering_results_ptr);
      nbr_offs_local += (info_local.num_vertices + 1);
      nbrs_local += (info_local.num_edges + 1);
      clustering_results_ptr += info_local.num_vertices;
    }
    
    for(size_t i=1; i<num_process; i++){
      MPI_Recv(num_cluster_total+i*local_n, local_n, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
      MPI_Recv(clustering_results_ptr, num_vertices[i], MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
      clustering_results_ptr += num_vertices[i];
    }

    free(info_edges);
    free(info_vertices);
    free(num_vertices);
      
  }
  else {
    int local_n = 0;
    int* local_info_edges = (int *)calloc(local_n, sizeof(int));
    int* local_info_vertices = (int *)calloc(local_n, sizeof(int));

    MPI_Recv(&local_n, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
    MPI_Recv(local_info_edges, local_n, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
    MPI_Recv(local_info_vertices, local_n, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

    int local_num_offs = 0;
    int local_num_nbrs = 0;
    for(size_t i=0; i<local_n; i++){
      local_num_offs += local_info_vertices[i] + 1;
      local_num_nbrs += local_info_edges[i] + 1;
    }

    int* local_nbr_offs = (int *)calloc(local_num_offs, sizeof(int));
    int* local_nbrs = (int *)calloc(local_num_nbrs, sizeof(int));
    MPI_Recv(local_nbr_offs, local_num_offs, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
    MPI_Recv(local_nbrs, local_num_nbrs, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

    int* local_clustering_results = (int *)calloc(local_num_offs-local_n, sizeof(int));
    int* local_num_cluster = (int *)calloc(local_n, sizeof(int));
    int* clustering_results_ptr = local_clustering_results;
    int* nbr_offs_ptr = local_nbr_offs;
    int* nbrs_ptr = local_nbrs;
    int total_num_vertices = 0;
    for(size_t i=0; i<local_n; i++){
      GraphMetaInfo info_local;
      info_local.num_edges = local_info_edges[i];
      info_local.num_vertices = local_info_vertices[i];

      local_num_cluster[i] = clustering(info_local, nbr_offs_ptr, 
                                nbrs_ptr, clustering_results_ptr);

      clustering_results_ptr += info_local.num_vertices;
      nbr_offs_ptr += info_local.num_vertices + 1;
      nbrs_ptr += info_local.num_edges + 1;
      total_num_vertices += info_local.num_vertices;
    }

    MPI_Send(local_num_cluster, local_n, MPI_INT, 0, 0, comm);
    MPI_Send(local_clustering_results, total_num_vertices, MPI_INT, 0, 0, comm);

    free(local_info_edges);
    free(local_info_vertices);
    free(local_nbr_offs);
    free(local_nbrs);
    free(local_clustering_results);
    free(local_num_cluster);
  }

  MPI_Barrier(comm);
  auto end_clock = chrono::high_resolution_clock::now();

  // 1) print results to screen
  if (my_rank == 0) {
    for (size_t i = 0; i < num_graphs; i++) {
      printf("num cluster in graph %d : %d\n", i, num_cluster_total[i]);
    }
    fprintf(stderr, "Elapsed Time: %.9lf ms\n",
            chrono::duration_cast<chrono::nanoseconds>(end_clock - start_clock)
                    .count() /
                pow(10, 6));
  }

  // 2) write results to file
  if (my_rank == 0) {
    int *result_graph = clustering_results;
    for (int i = 0; i < num_graphs; i++) {
      GraphMetaInfo info_local = info[i];
      write_result_to_file(info_local, i, num_cluster_total[i], result_graph,
                           result_path);

      result_graph += info_local.num_vertices;
    }
  }

  MPI_Finalize();

  if (my_rank == 0) {
    free(num_cluster_total);
  }

  return 0;
}
