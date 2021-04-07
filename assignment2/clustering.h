#pragma once

#include <cmath>
#include <cstring>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;


void read_file(string file_path, int& num_vs, int& num_es, int*& nbr_offs, int*& nbrs) {
    std::ifstream inputf(file_path, std::ifstream::in);
    char sharp;
    inputf >> sharp >> num_vs >> num_es;
    nbr_offs = new int[num_vs + 1];
    nbrs = new int[num_es + 1];

    int cur_off = 0;
    int last_v_id = -1;

    for (int i = 0; i < num_es; i++) {
        int v_id, nbr_id;
        inputf >> v_id >> nbr_id;
        if (v_id != last_v_id) {
            nbr_offs[v_id] = cur_off;
            last_v_id = v_id;
        }
        nbrs[cur_off++] = nbr_id;
    }
    assert(last_v_id == num_vs - 1 && cur_off == num_es && "Input file format error!");
    nbr_offs[num_vs] = cur_off;
}


int get_num_com_nbrs(int *left_start, int *left_end, int *right_start, int *right_end) {
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

void write_result_to_file(int nv, int num_cluster, int *res, string &path) {

    ofstream fs(path);
    fs << num_cluster << endl;

    for (int i = 0; i < nv; i++) {
        fs << res[i] << ' ';
    }
    fs << endl;
    fs.close();
}
