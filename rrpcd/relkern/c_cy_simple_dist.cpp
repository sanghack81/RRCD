#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include <random>
#include <cmath>
#include <vector>
#include <iostream>

double r_convolution00(double *v1s, double *v2s, int vlen1, int vlen2, double gamma, int equal_size_only) {
    if (equal_size_only > 0 && vlen1 != vlen2) {
        return 0.0;
    }
    if (vlen1 == vlen2 && vlen1 == 0) {
        return 1.0;
    }
    int i, j;
    double v = 0.0;
    for (i = 0; i < vlen1; i++) {
        for (j = 0; j < vlen2; j++) {
            v += exp(-gamma * pow(v1s[i] - v2s[j], 2.0));
        }
    }
    return v;
}

void c_sub_relational_kernel00(double *values, int *lengths, double *output, int n, int divide, int remnant, double gamma, int equal_size_only) {
    double *value_i = values;

    int k = 0;
    for (int i = 0; i < n; i++) {
        double *value_j = value_i;
        for (int j = i; j < n; j++, k++) {
            if (k % divide == remnant) { // thread-based computation
                output[j * n + i] = output[i * n + j] = r_convolution00(value_i, value_j, lengths[i], lengths[j], gamma, equal_size_only);
            }

            value_j += lengths[j];
        }
        value_i += lengths[i];
    }
}

void c_relational_kernel00(double *values, int *lengths, double *output, int n, int n_threads, double gamma, int equal_size_only) {
    if (n_threads == 1) {
        c_sub_relational_kernel00(values, lengths, output, n, n_threads, 0, gamma, equal_size_only);
        std::cout << ""; // TODO strange behavior? output unflushed?
    } else {
        std::vector<std::thread> threads;
        for (int i = 0; i < n_threads; i++) {
            threads.push_back(std::thread(c_sub_relational_kernel00, values, lengths, output, n, n_threads, i, gamma, equal_size_only));
        }
        // joint threads,
        for (int i = 0; i < n_threads; i++) {
            threads[i].join();
        }
        std::cout << ""; // TODO strange behavior?
    }
}
