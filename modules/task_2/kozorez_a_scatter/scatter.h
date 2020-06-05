// Copyright 2020 Kozorez Alexandr
#ifndef MODULES_TASK_2_KOZOREZ_A_SCATTER_SCATTER_H_
#define MODULES_TASK_2_KOZOREZ_A_SCATTER_SCATTER_H_
#include <mpi.h>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <random>
#include <ctime>

int myScatter(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                       void* recvbuf, int recvcount, MPI_Datatype recvtype,
                       int root, MPI_Comm comm);

double sumOfVector(std::vector<int> vec);

void getRandomVector(int* ptr, int size);

#endif  // MODULES_TASK_2_KOZOREZ_A_SCATTER_SCATTER_H_
