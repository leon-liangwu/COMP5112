#pragma once

#include <cmath>
#include <cstring>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

void read_file(std::string file_path, int& num_vs, int& num_es, int*& nbr_offs, int*& nbrs);

void write_result_to_file(int nv, int num_cluster, int *res, std::string &path);
