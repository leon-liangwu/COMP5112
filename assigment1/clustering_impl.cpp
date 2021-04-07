#include "clustering.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

int get_num_com_nbrs(int *left_start, int *left_end, int *right_start,
                     int *right_end) {
  int *left_pos = left_start, *right_pos = right_start, num_com_nbrs = 0;

  while (left_pos < left_end && right_pos < right_end) {
    if (*left_pos == *right_pos) {
      num_com_nbrs++;
      left_pos++;
      right_pos++;
    } else if (*left_pos < *right_pos) {
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
    if (visited[nbr_id])
      continue;

    visited[nbr_id] = true;
    cluster_result[nbr_id] = num_clusters;
    if (pivots[nbr_id]) {
      expansion(nbr_id, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots,
                cluster_result);
    }
  }
}

int clustering(GraphMetaInfo &info, int *nbr_offs, int *nbrs, int *res,
               float epsilon, int mu) {

  int num_vs = info.num_vertices;
  int num_es = info.num_edges;

  bool *pivots = new bool[num_vs]();
  int *num_sim_nbrs = new int[num_vs]();
  int **sim_nbrs = new int *[num_vs];

  for (int i = 0; i < num_vs; i++) {
    int *left_start = &nbrs[nbr_offs[i]];   // start ptr for nbr of v[i]
    int *left_end = &nbrs[nbr_offs[i + 1]]; // end ptr for nbr of v[i]
    int left_size = left_end - left_start;

    sim_nbrs[i] = new int[left_size];
    // loop over all neighbors of i
    for (int *j = left_start; j < left_end; j++) {
      int nbr_id = *j;

      int *right_start = &nbrs[nbr_offs[nbr_id]];
      int *right_end = &nbrs[nbr_offs[nbr_id + 1]];
      int right_size = right_end - right_start;

      // compute the similarity
      int num_com_nbrs =
          get_num_com_nbrs(left_start, left_end, right_start, right_end);

      float sim = (num_com_nbrs + 2) /
                  std::sqrt((left_size + 1.0) * (right_size + 1.0));

      if (sim > epsilon) {
        sim_nbrs[i][num_sim_nbrs[i]] = nbr_id;
        num_sim_nbrs[i]++;
      }
    }
    if (num_sim_nbrs[i] > mu)
      pivots[i] = true;
  }

  bool *visited = new bool[num_vs]();
  // int *cluster_result = new int[num_vs];
  std::fill(res, res + num_vs, -1);
  int num_clusters = 0;
  for (int i = 0; i < num_vs; i++) {
    if (!pivots[i] || visited[i])
      continue;

    visited[i] = true;
    res[i] = num_clusters;
    expansion(i, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots, res);

    num_clusters++;
  }

  delete[] pivots;
  delete[] num_sim_nbrs;
  delete[] sim_nbrs;
  delete[] visited;

  return num_clusters;
}

int read_files(std::string &dir, GraphMetaInfo *&info, int *&nbr_offs,
               int *&nbrs) {
  std::ifstream meta(dir + '/' + "meta", std::ifstream::in);

  int num_graph;
  meta >> num_graph;

  int num_vs, num_es;
  int num_vs_total = 0, num_es_total = 0;
  info = (GraphMetaInfo *)calloc(num_graph, sizeof(GraphMetaInfo));
  for (int i = 0; i < num_graph; i++) {
    std::string file_path = dir + '/' + std::to_string(i) + ".txt";
    std::ifstream inputf(file_path, std::ifstream::in);

    char sharp;
    inputf >> sharp >> num_vs >> num_es;
    info[i].num_vertices = num_vs;
    info[i].num_edges = num_es;

    num_vs_total += num_vs + 1;
    num_es_total += num_es + 1;
  }

  nbr_offs = (int *)calloc(num_vs_total, sizeof(int));
  nbrs = (int *)calloc(num_es_total, sizeof(int));

  int *nbr_offs_curr = nbr_offs;
  int *nbrs_curr = nbrs;

  for (int i = 0; i < num_graph; i++) {
    std::string file_path = dir + '/' + std::to_string(i) + ".txt";
    std::ifstream inputf(file_path, std::ifstream::in);

    char sharp;
    inputf >> sharp >> num_vs >> num_es;
    // cout << num_vs << ' ' << num_es << endl;

    int cur_off = 0;
    int last_v_id = -1;

    for (int i = 0; i < num_es; i++) {
      int v_id, nbr_id;
      inputf >> v_id >> nbr_id;

      // edges starting with a new vertex
      if (v_id != last_v_id) {
        nbr_offs_curr[v_id] = cur_off;
        last_v_id = v_id;
      }
      nbrs_curr[cur_off++] = nbr_id;
    }

    assert(last_v_id == num_vs - 1 && cur_off == num_es &&
           "Input file format error!");

    nbr_offs_curr[num_vs] = cur_off;
    nbr_offs_curr += (num_vs + 1);
    nbrs_curr += (num_es + 1);
  }

  return num_graph;
}

void write_result_to_file(GraphMetaInfo &info, int idx, int num_cluster,
                          int *res, std::string &path) {
  int nv = info.num_vertices;

  ofstream fs(path + '/' + to_string(idx) + ".txt");
  fs << num_cluster << endl;

  for (int i = 0; i < info.num_vertices; i++) {
    fs << res[i] << ' ';
  }
  fs << endl;
  fs.close();
}