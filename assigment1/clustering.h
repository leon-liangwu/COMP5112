#pragma once

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

struct GraphMetaInfo {
  int num_edges = 0;
  int num_vertices = 0;
};

/**
 * @brief read graph information from dataset
 *
 * @param dir       dataset path
 * @param info      graph information
 * @param nbr_offs  offsets for each vertex
 * @param nbrs      neighbor indices
 *
 * @return int      number of graphs
 */
int read_files(std::string &dir, GraphMetaInfo *&info, int *&nbr_offs,
               int *&nbrs);

/**
 * @brief Get the num com nbrs object
 * 
 * @param left_start 
 * @param left_end 
 * @param right_start 
 * @param right_end 
 * @return int 
 */
int get_num_com_nbrs(int *left_start, int *left_end, int *right_start,
                     int *right_end);

/**
 * @brief 
 * 
 * @param cur_id 
 * @param num_clusters 
 * @param num_sim_nbrs 
 * @param sim_nbrs 
 * @param visited 
 * @param pivots 
 * @param cluster_result 
 */
void expansion(int cur_id, int num_clusters, int *num_sim_nbrs, int **sim_nbrs,
               bool *visited, bool *pivots, int *cluster_result);

/**
 * @brief perform clustering in a single graph with SCAN
 *
 * @param info      information of graph
 * @param nbr_offs  offsets for neighbor data
 * @param nbrs      neighbor data
 * @param res       clustering result array
 * @param eps       parameter for algorithm
 * @param mu        parameter for algorithm
 *
 * @return int result
 */
int clustering(GraphMetaInfo &info, int *nbr_offs, int *nbrs, int *res,
               float eps = 0.7, int mu = 3);

void write_result_to_file(GraphMetaInfo &info, int idx, int num_cluster,
                          int *res, std::string &path);
