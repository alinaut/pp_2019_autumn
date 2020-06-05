// Copyright 2020 Kozorez Alexandr
#include <mpi.h>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include "./scatter.h"

int myScatter(void* send_buf, int send_count, MPI_Datatype send_type,
                       void* recv_buf, int recv_count, MPI_Datatype recv_type,
                       int root, MPI_Comm comm) {
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int send_type_size, recv_type_size;

  if (MPI_Type_size(send_type, &send_type_size) == MPI_ERR_TYPE)
    return MPI_ERR_TYPE;

  if (MPI_Type_size(recv_type, &recv_type_size) == MPI_ERR_TYPE)
    return MPI_ERR_TYPE;

  if (send_count != recv_count || send_count <= 0 || recv_count <= 0 ||
      root < 0)
    return MPI_ERR_COUNT;

  if (rank == root) {
    memcpy(recv_buf,
           static_cast<char*>(send_buf) + rank * send_count * send_type_size,
           send_count * send_type_size);

    for (int i = 0; i < size; i++) {
      if (i == root) continue;
      MPI_Send(static_cast<char*>(send_buf) + i * send_count * send_type_size,
               send_count, send_type, i, 0, comm);
    }
  } else {
    MPI_Status status;
    MPI_Recv(recv_buf, recv_count, recv_type, root, 0, comm, &status);
  }
  return MPI_SUCCESS;
}

double sumOfVector(std::vector<int> vec) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int vec_size = 0;
  if (rank == 0) vec_size = vec.size();

  MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int chunk = vec_size / size;
  int ost = vec_size % size;

  std::vector<double> local_vec(chunk);

  if (rank == 0) local_vec.resize(chunk + ost);

  MPI_Scatter(&vec[0], chunk, MPI_INT, &local_vec[0], chunk, MPI_INT, 0,
              MPI_COMM_WORLD);
  if (rank == 0) {
    for (int i = 0; i < chunk + ost; ++i) local_vec[i] = vec[i];
  }

  int local_size = local_vec.size();
  double global_sum = 0;
  double local_sum = 0;
  for (int i = 0; i < local_size; ++i) local_sum += local_vec[i];

  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  return global_sum;
}

void getRandomVector(int* ptr, int size) {
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  for (int i = 0; i < size; i++) {
    ptr[i] = gen() % 100;
  }
}
