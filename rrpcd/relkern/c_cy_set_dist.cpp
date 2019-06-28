#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include <random>
#include <cmath>
#include <vector>
#include "Matrix.h"
#include "Hungarian.h"


double
r_convolution(double *v1s, double *v2s, int vlen1, int vlen2, int *item1s, int *item2s, double *vertex_kernel, int m,
              double gamma, int equal_size_only) {
    if (equal_size_only > 0 && vlen1 != vlen2) {
        return 0.0;
    }
    int i, j;
    double v = 0.0;
    for (i = 0; i < vlen1; i++) {
        for (j = 0; j < vlen2; j++) {
            v += vertex_kernel[item1s[i] * m + item2s[j]] * exp(-gamma * pow(v1s[i] - v2s[j], 2.0));
        }
    }
    return v;
}

double
optimal_assignment(double *v1s, double *v2s, int vlen1, int vlen2, int *item1s, int *item2s, double *vertex_kernel,
                   int m, double gamma, int equal_size_only) {
    int i, j;
    double val;
    if (equal_size_only > 0 && vlen1 != vlen2) {
        return 0.0;
    }
    if (vlen1 == 0 && vlen2 == 0) {
        return 1.0;
    }
    if (vlen1 == 0 || vlen2 == 0) {
        return 0.0;
    }
    if (vlen1 == 1 && vlen2 == 1) {
        return vertex_kernel[item1s[0] * m + item2s[0]] * exp(-gamma * pow(v1s[0] - v2s[0], 2.0));
    }
    if (vlen1 == 1) {
        double max_val = 0.0;
        for (int k = 0; k < vlen2; k++) {
            double temp_val = vertex_kernel[item1s[0] * m + item2s[k]] * exp(-gamma * pow(v1s[0] - v2s[k], 2.0));
            if (max_val < temp_val) {
                max_val = temp_val;
            }
        }
        return max_val;
    }
    if (vlen2 == 1) {
        double max_val = 0.0;
        for (int k = 0; k < vlen1; k++) {
            double temp_val = vertex_kernel[item1s[k] * m + item2s[0]] * exp(-gamma * pow(v1s[k] - v2s[0], 2.0));
            if (max_val < temp_val) {
                max_val = temp_val;
            }
        }
        return max_val;
    }
    if (vlen1 <= 4 && vlen2 <= 4) {
        std::vector<double> storage(16, 0.0);
        for (i = 0; i < vlen1; i++) {
            for (j = 0; j < vlen2; j++) {
                storage[i * 4 + j] = vertex_kernel[item1s[i] * m + item2s[j]] * exp(-gamma * pow(v1s[i] - v2s[j], 2.0));
            }
        }
        if (vlen1 <= 2 && vlen2 <= 2) {
            return max(storage[0 * 4 + 0] + storage[1 * 4 + 1], storage[1 * 4 + 0] + storage[0 * 4 + 1]);
        }
        if (vlen1 <= 3 && vlen2 <= 3) {
            double max_val = 0.0;
            max_val = max(max_val, storage[0 * 4 + 0] + storage[1 * 4 + 1] + storage[2 * 4 + 2]);
            max_val = max(max_val, storage[0 * 4 + 0] + storage[1 * 4 + 2] + storage[2 * 4 + 1]);
            max_val = max(max_val, storage[0 * 4 + 1] + storage[1 * 4 + 0] + storage[2 * 4 + 2]);
            max_val = max(max_val, storage[0 * 4 + 1] + storage[1 * 4 + 2] + storage[2 * 4 + 0]);
            max_val = max(max_val, storage[0 * 4 + 2] + storage[1 * 4 + 0] + storage[2 * 4 + 1]);
            return max(max_val, storage[0 * 4 + 2] + storage[1 * 4 + 1] + storage[2 * 4 + 0]);
        }
        double max_val = 0.0;
        if (vlen1 <= 4 && vlen2 <= 4) {
            for (i = 0; i < 4; i++) {
                for (j = 0; j < 4; j++) {
                    if (i == j) {
                        continue;
                    }
                    for (int k = 0; k < 4; k++) {
                        if (k == i || k == j) {
                            continue;
                        }
                        // i+j+k+l == 0+1+2+3 == 6
                        int l = 6 - i - j - k;
                        val = storage[0 * 4 + i] + storage[1 * 4 + j] + storage[2 * 4 + k] + storage[3 * 4 + l];
                        if (max_val < val) {
                            max_val = val;
                        }
                    }
                }
            }
            return max_val;
        }
    }


    // treat storage as (vlen1, vlen2) matrix
    Matrix matrix;
    if (vlen1 <= vlen2) { // (vlen1, vlen2) matrix
        matrix.resize(vlen1);
        for (i = 0; i < vlen1; i++) {
            matrix[i].resize(vlen2);
            for (j = 0; j < vlen2; j++) {
                val = vertex_kernel[item1s[i] * m + item2s[j]] * exp(-gamma * pow(v1s[i] - v2s[j], 2.0));
                matrix[i][j].SetWeight(val);
            }
        }
    } else { // (vlen2, vlen1) matrix
        matrix.resize(vlen2);
        for (j = 0; j < vlen2; j++) {
            matrix[j].resize(vlen1);
            for (i = 0; i < vlen1; i++) {
                matrix[j][i].SetWeight(
                        vertex_kernel[item1s[i] * m + item2s[j]] * exp(-gamma * pow(v1s[i] - v2s[j], 2.0)));
            }
        }
    }
    // TODO faster
    // (maximize) linear sum assignment
    BipartiteGraph bg(matrix);
    Hungarian h(bg);
    h.HungarianAlgo();
    return h.OptimalValue();
}


void c_sub_relational_kernel(double *values, int *items, int *lengths, double *vertex_kernel,
              double *output,
              int n,
              int m, int divide, int remnant, double gamma, int algorithm, int equal_size_only) {
//    cout << "";
    double *value_i = values;
    int *items_i = items;

    for (int i = 0; i < n; i++) {
        double *value_j = value_i;
        int *items_j = items_i;
        for (int j = i; j < n; j++) {
            if ((i + j) % divide == remnant) { // thread-based computation
                if (algorithm == 0) {
                    output[j * n + i] = output[i * n + j] = r_convolution(value_i, value_j,
                                                                          lengths[i], lengths[j],
                                                                          items_i, items_j,
                                                                          vertex_kernel, m, gamma, equal_size_only);
                } else if (algorithm == 1) {
                    output[j * n + i] = output[i * n + j] = optimal_assignment(value_i, value_j,
                                                                               lengths[i], lengths[j],
                                                                               items_i, items_j,
                                                                               vertex_kernel, m, gamma,
                                                                               equal_size_only);
                } else {
                    output[j * n + i] = output[i * n + j] = 0.0;
                }

            }

            value_j += lengths[j];
            items_j += lengths[j];
        }
        value_i += lengths[i];
        items_i += lengths[i];
    }
}

void c_relational_kernel(double *values, int *items, int *lengths, double *vertex_kernel,
          double *output,
          int n,
          int m, int n_threads, double gamma, int algorithm, int equal_size_only) {
    //
    // values: unrolled values, [pair[1] for tuple in data for pair in tuple]
    // items: unrolled items    [pair[0] for tuple in data for pair in tuple]
    // lengths:                 [len(tuple) for tuple in data]
    // vertex_kernel: unrolled vertex kernel matrix, of the shape (m,m)
    // output: output kernel matrix, must be of the shape, (n, n)
    // n: number of instances of data,
    // m,
    if (n_threads == 1) {
        c_sub_relational_kernel(values, items, lengths, vertex_kernel, output, n, m, n_threads, 0, gamma, algorithm, equal_size_only);
        cout << ""; // TODO strange behavior? output unflushed?
    } else {
        std::vector<std::thread> threads;
        for (int i = 0; i < n_threads; i++) {
            threads.push_back(std::thread(c_sub_relational_kernel, values, items, lengths, vertex_kernel,
                                          output, n, m, n_threads, i, gamma, algorithm, equal_size_only));
        }
        // joint threads,
        for (int i = 0; i < n_threads; i++) {
            threads[i].join();
        }
        cout << ""; // TODO strange behavior?
    }
}
