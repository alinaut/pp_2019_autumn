// Copyright 2020 Kozorez Alexandr
#ifndef MODULES_TASK_3_KOZOREZ_A_GAUSS_GAUSS_H_
#define MODULES_TASK_3_KOZOREZ_A_GAUSS_GAUSS_H_

struct rgb {
    unsigned char red, green, blue;
};

const int Mask[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

const int sumOfElementsInMask = 16;

rgb* createImageOne(int n, int m);
rgb* createImageTwo(int n, int m);

class GaussianBlurs {
 public:
    rgb *source;
    rgb *result;
    rgb Processing(int i, int j);
    int _cols;
    int _rows;
    int a;
    int b;
    GaussianBlurs(rgb* input, int rows, int cols);
    rgb* GetResult();
    void process();
};

rgb* linear_filter_with_gauss(int n, int m, int code = 0);


#endif  // MODULES_TASK_3_KOZOREZ_A_GAUSS_GAUSS_H_
