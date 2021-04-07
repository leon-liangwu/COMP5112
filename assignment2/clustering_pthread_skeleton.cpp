/*
 * Name: WU Liang
 * Student id: 20636843
 * ITSC email: lwuat@connect.ust.hk
 *
 * Please only change this file and do not change any other files.
 * Feel free to change/add any helper functions.
 *
 * COMPILE: g++ -lstdc++ -std=c++11 -lpthread clustering_pthread_skeleton.cpp main.cpp -o pthread
 * RUN:     ./pthread <path> <epsilon> <mu> <num_threads>
 */

#include <pthread.h>
#include <semaphore.h>
#include "clustering.h"
#include <unistd.h>

int g_num_vs;
float g_epsilon;
int g_mu;
int *g_nbr_offs;
int *g_nbrs;
int *g_cluster_result;

// stage 1 vals
bool *pivots;
int *num_sim_nbrs;
int **sim_nbrs;

// stage 2 vals
bool *visited;

int counter;
sem_t count_sem;
sem_t barrier_sem;

pthread_mutex_t *mutex;

struct AllThings{
    int num_threads;
    int my_rank;

    AllThings(int inum_threads, int imy_rank){
        num_threads = inum_threads;
        my_rank = imy_rank;
    };
};

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

void *parallel(void* allthings){
    AllThings *all = (AllThings *) allthings;

//    printf("Hello from %d of %d\n", all->my_rank, all->num_threads);
    int num_threads = all->num_threads;
    int my_rank = all->my_rank;

    int my_n = g_num_vs/num_threads;
    int my_first_i = my_n*my_rank;
    int my_last_i = my_first_i + my_n;

    int i, j;

    // stage 1
    for(i=my_first_i; i<my_last_i; i++){
        int *left_start = &g_nbrs[g_nbr_offs[i]];
        int *left_end = &g_nbrs[g_nbr_offs[i + 1]];
        int left_size = left_end - left_start;

        sim_nbrs[i] = new int[left_size];
        // loop over all neighbors of i
        for (int *j = left_start; j < left_end; j++) {
            int nbr_id = *j;

            int *right_start = &g_nbrs[g_nbr_offs[nbr_id]];
            int *right_end = &g_nbrs[g_nbr_offs[nbr_id + 1]];
            int right_size = right_end - right_start;

            // compute the similarity
            int num_com_nbrs = get_num_com_nbrs(left_start, left_end, right_start, right_end);

            float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

            if (sim > g_epsilon) {
                sim_nbrs[i][num_sim_nbrs[i]] = nbr_id;
                num_sim_nbrs[i]++;
            }
        }
        if (num_sim_nbrs[i] > g_mu) pivots[i] = true;
    }

    // barrier
    sem_wait(&count_sem);
    if (counter == num_threads - 1) {
        counter = 0;
        sem_post(&count_sem);
        for (j = 0; j < num_threads-1; j++)
            sem_post(&barrier_sem);

    } else {
        counter++;
        sem_post(&count_sem);
        sem_wait(&barrier_sem);
    }

    // stage 2
    int num_clusters = 0;
    for (int i = my_first_i; i < my_last_i; i++) {
        pthread_mutex_lock(&mutex[i]);
        if (!pivots[i] || visited[i]) continue;
        visited[i] = true;
        pthread_mutex_unlock(&mutex[i]);
        g_cluster_result[i] = i;
        expansion(i, i, num_sim_nbrs, sim_nbrs, visited, pivots, g_cluster_result);

        num_clusters++;
    }

    return 0;
}

int *scan(float epsilon, int mu, int num_threads, int num_vs, int num_es, int *nbr_offs, int *nbrs){
    long thread;
    pthread_t* thread_handles = (pthread_t*) malloc(num_threads*sizeof(pthread_t));
    int *cluster_result = new int[num_vs];
    std::fill(cluster_result, cluster_result + num_vs, -1);

    counter = 0;
    sem_init(&barrier_sem, 0, 0);
    sem_init(&count_sem, 0, 1);

    mutex = new pthread_mutex_t[num_vs]();

    g_cluster_result = cluster_result;
    g_num_vs = num_vs;
    g_epsilon = epsilon;
    g_mu = mu;
    g_nbr_offs = nbr_offs;
    g_nbrs = nbrs;

    pivots = new bool[num_vs]();
    num_sim_nbrs = new int[num_vs]();
    sim_nbrs = new int*[num_vs];

    visited = new bool[num_vs]();

    for (thread=0; thread < num_threads; thread++)
        pthread_create(&thread_handles[thread], NULL, parallel, (void *) new AllThings(
                num_threads, thread));
    for (thread=0; thread < num_threads; thread++)
        pthread_join(thread_handles[thread], NULL);

    return cluster_result;
}

