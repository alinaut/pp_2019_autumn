// Copyright 2020 Kozorez Alexandr
#define DIFFERENT_SIZE "Matrix and vector has different size"
#define NEGATIVE_SIZE "Matrix has negative size"
#define BIG_SIZE "Matrix size is too big"

#include <mpi.h>
#include <random>
#include <vector>
#include <stdexcept>
#include <ctime>
#include "./matrix_sum.h"

int getParallelSum(std::vector<int> matrix, int n, int m) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int chunk, index;
    int local_sum, global_sum = 0;
    if (rank == 0) {
        if (m * n != static_cast<int>(matrix.size())) {
            int err = -1;
            for (int proc_num = 1; proc_num < size; proc_num++)
                MPI_Send(&err, 1, MPI_INT, proc_num, 0, MPI_COMM_WORLD);
            throw std::runtime_error(DIFFERENT_SIZE);
        }
        int send_size = 0;
        chunk = n * m / size;
        int ostatok = (n * m) % size;
        index = chunk;
        for (int proc_num = 1; proc_num < size; proc_num++) {
            if (ostatok > 0) {
                ostatok--;
                send_size = chunk + 1;
            } else {
                send_size = chunk;
            }
            MPI_Send(&send_size, 1, MPI_INT, proc_num, 0, MPI_COMM_WORLD);
            MPI_Send(&matrix[0]+index, send_size, MPI_INT, proc_num, 1, MPI_COMM_WORLD);
            index+=send_size;
        }
        local_sum = getSequentialSum(std::vector<int>(matrix.begin(), matrix.begin()+chunk));
    } else {
        MPI_Status status;
        MPI_Recv(&chunk, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (chunk == -1)
            throw std::runtime_error(DIFFERENT_SIZE);
        std::vector<int> local_vec(chunk);
        MPI_Recv(&local_vec[0], chunk, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        local_sum = getSequentialSum(local_vec);
    }
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    return global_sum;
}

int getSequentialSum(std::vector<int> vec) {
    int sum = 0;
    int size = vec.size();
    for (int i = 0; i < size; i++)
        sum+=vec[i];
    return sum;
}

std::vector<int> getRandomMatrix(int n, int m) {
    if (n <= 0 || m <= 0)
        throw std::runtime_error(NEGATIVE_SIZE);
    if (n > (INT32_MAX/100)/m || m > (INT32_MAX/100)/n)
        throw std::runtime_error(BIG_SIZE);
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> matrix(n*m);
    for (int i = 0; i < n*m; i++)
        matrix[i] = gen() % 100;
    return matrix;
}
