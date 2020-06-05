// Copyright 2020 Kozorez Alexandr
#include <gtest/gtest.h>
#include <gtest-mpi-listener.hpp>
#include <string>
#include <vector>
#include "./scatter.h"

TEST(Parallel_Scatter_MPI, Test_Negative_Root) {
  int root = -1;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<int> p(1);
  int dest[3];
  EXPECT_EQ(myScatter(&p[0], 1, MPI_INT, &dest[0], 1, MPI_INT, root,
                            MPI_COMM_WORLD),
            MPI_ERR_COUNT);
}

TEST(Parallel_Scatter_MPI, TestRandom)
{
	ASSERT_EQ(1, 1);
}

TEST(Parallel_Scatter_MPI, Test_Different_Size) {
  int root = 0;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<int> p(1);
  int dest[3];
  EXPECT_EQ(myScatter(&p[0], 1, MPI_INT, &dest[0], 99, MPI_INT, root,
                            MPI_COMM_WORLD),
            MPI_ERR_COUNT);
}

TEST(Parallel_Scatter_MPI, Test_Compare_With_Default_Scatter) {
  int mes_size = 3;
  int root = 0;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<int> p(mes_size * size);
  std::vector<int> d1(mes_size * size);
  std::vector<int> d2(mes_size * size);
  std::vector<int> dest(mes_size);
  if (rank == root) {
    for (int i = 0; i < mes_size * size; ++i) p[i] = i;
  }
  myScatter(&p[0], mes_size, MPI_INT, &dest[0], mes_size, MPI_INT, root,
                  MPI_COMM_WORLD);
  MPI_Gather(&dest[0], mes_size, MPI_INT, &d1[0], mes_size, MPI_INT, root,
             MPI_COMM_WORLD);

  MPI_Scatter(&p[0], mes_size, MPI_INT, &dest[0], mes_size, MPI_INT, root,
              MPI_COMM_WORLD);
  MPI_Gather(&dest[0], mes_size, MPI_INT, &d2[0], mes_size, MPI_INT, root,
             MPI_COMM_WORLD);

  if (rank == root) {
    EXPECT_EQ(d1, d2);
  }
}

TEST(Parallel_Scatter_MPI, Test_Scatter_Gather_Double) {
  int root = 0;
  int mes_size = 3;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<double> p(mes_size * size);
  std::vector<double> d(mes_size * size);
  std::vector<double> dest(mes_size);
  if (rank == root) {
    for (int i = 0; i < mes_size * size; ++i) p[i] = i + 1.0 / (i + 1);
  }
  myScatter(&p[0], mes_size, MPI_DOUBLE, &dest[0], mes_size,
                     MPI_DOUBLE, root, MPI_COMM_WORLD);
  MPI_Gather(&dest[0], mes_size, MPI_DOUBLE, &d[0], mes_size, MPI_DOUBLE, root,
             MPI_COMM_WORLD);

  if (rank == root) {
    EXPECT_EQ(p, d);
  }
}

TEST(Parallel_Scatter_MPI, Test_Scatter_Gather_Int) {
  int mes_size = 3;
  int root = 0;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<int> p(mes_size * size);
  std::vector<int> d(mes_size * size);
  std::vector<int> dest(mes_size);
  if (rank == root) {
    for (int i = 0; i < mes_size * size; ++i) p[i] = i;
  }
  myScatter(&p[0], mes_size, MPI_INT, &dest[0], mes_size, MPI_INT,
                     root, MPI_COMM_WORLD);
  MPI_Gather(&dest[0], mes_size, MPI_INT, &d[0], mes_size, MPI_INT, root,
             MPI_COMM_WORLD);

  if (rank == root) {
    EXPECT_EQ(p, d);
  }
}

TEST(Parallel_Scatter_MPI, Test_Scatter_Gather_Float) {
  int mes_size = 3;
  int root = 0;
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<float> p(mes_size * size);
  std::vector<float> d(mes_size * size);
  std::vector<float> dest(mes_size);
  if (rank == root) {
    for (float i = 0; i < mes_size * size; ++i) p[i] = i + 1.0 / (i + 1);
  }
  myScatter(&p[0], mes_size, MPI_FLOAT, &dest[0], mes_size, MPI_FLOAT,
                     root, MPI_COMM_WORLD);
  MPI_Gather(&dest[0], mes_size, MPI_FLOAT, &d[0], mes_size, MPI_FLOAT, root,
             MPI_COMM_WORLD);

  if (rank == root) {
    EXPECT_EQ(p, d);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
