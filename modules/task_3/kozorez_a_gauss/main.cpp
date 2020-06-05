// Copyright 2020 Kozorez Alexandr
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./gauss.h"

TEST(Parallel_Gauss_MPI, Test_Processing) {
    int n = 3, m = 9;
    rgb* input = createImageTwo(n, m);
    GaussianBlurs* gb = new GaussianBlurs(input, n, m);
    ASSERT_NO_THROW(gb -> process());
}

TEST(Parallel_Gauss_MPI, Test_ImageOne_10x10) {
    int n = 10, m = 10;
    rgb* result = linear_filter_with_gauss(n, m);
    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (static_cast<int>(result[i*m + j].red) < 0 || static_cast<int>(result[i*m + j].red) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].green) < 0 || static_cast<int>(result[i*m + j].green) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].blue) < 0 || static_cast<int>(result[i*m + j].blue) > 256) {
                ans = ans + 1;
            }
        }
    }
    EXPECT_EQ(ans, 0);
}

TEST(Parallel_Gauss_MPI, Test_10x5) {
    int n = 10, m = 5;
    rgb* result = linear_filter_with_gauss(n, m);
    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (static_cast<int>(result[i*m + j].red) < 0 || static_cast<int>(result[i*m + j].red) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].green) < 0 || static_cast<int>(result[i*m + j].green) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].blue) < 0 || static_cast<int>(result[i*m + j].blue) > 256) {
                ans = ans + 1;
            }
        }
    }
    EXPECT_EQ(ans, 0);
}


TEST(Parallel_Gauss_MPI, Test_5x10) {
    int n = 5, m = 10;
    rgb* result = linear_filter_with_gauss(n, m);
    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (static_cast<int>(result[i*m + j].red) < 0 || static_cast<int>(result[i*m + j].red) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].green) < 0 || static_cast<int>(result[i*m + j].green) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].blue) < 0 || static_cast<int>(result[i*m + j].blue) > 256) {
                ans = ans + 1;
            }
        }
    }
    EXPECT_EQ(ans, 0);
}

TEST(Parallel_Gauss_MPI, Test_ImageTwo_10x10) {
    int n = 10, m = 10;
    rgb* result = linear_filter_with_gauss(n, m, 1);
    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (static_cast<int>(result[i*m + j].red) < 0 || static_cast<int>(result[i*m + j].red) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].green) < 0 || static_cast<int>(result[i*m + j].green) > 256) {
                ans = ans + 1;
            }
            if (static_cast<int>(result[i*m + j].blue) < 0 || static_cast<int>(result[i*m + j].blue) > 256) {
                ans = ans + 1;
            }
        }
    }
    EXPECT_EQ(ans, 0);
}

TEST(Parallel_Gauss_MPI, Test_linear_Parallel) {
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 3, m = 3;
    rgb *input = new rgb[n*m];
    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            input[k].red = 128;
            input[k].green = 128;
            input[k].blue = 128;
            ++k;
        }
    }
    rgb* result = linear_filter_with_gauss(n, m);
    if (rank == 0) {
        int ans = 0;
	    for (int i = 0; i < n; i++) {
	        for (int j = 0; j < m; j++) {
	            if (static_cast<int>(result[i*m + j].red) < 0 || static_cast<int>(result[i*m + j].red) > 256) {
	                ans = ans + 1;
	            }
	            if (static_cast<int>(result[i*m + j].green) < 0 || static_cast<int>(result[i*m + j].green) > 256) {
	                ans = ans + 1;
	            }
	            if (static_cast<int>(result[i*m + j].blue) < 0 || static_cast<int>(result[i*m + j].blue) > 256) {
	                ans = ans + 1;
	            }
	        }
	    }
	    EXPECT_EQ(ans, 0);
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
