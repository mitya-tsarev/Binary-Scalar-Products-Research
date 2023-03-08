#define ARMA_DONT_USE_STD_MUTEX
#include <iostream>
#include <vector>
#include <chrono>
#include <armadillo>
#include <boost/graph/adjacency_list.hpp>
#include "cube-matrices.h"

template<typename T>
std::vector<arma::Mat<T>> all_sym_invertible(int32_t dim) {
    std::vector<arma::Mat<T>> sym_inv;
    int32_t param_length = (dim * (dim + 1)) / 2;
    int64_t mat_number = 1 << param_length;
    for (int64_t ind = 0; ind < mat_number; ++ind) {
        arma::Mat<T> m(dim, dim);
        int32_t digit = 0;
        for (int32_t row = 0; row < dim; ++row) {
            for (int32_t col = row; col < dim; ++col) {
                m(row, col) = m(col, row) = T((ind >> digit) & 1);
                ++digit;
            }
        }
        if (arma::rank(arma::conv_to<arma::mat>::from(m)) == dim) {
            sym_inv.push_back(m);
        }
    }
    return sym_inv;
}

template<typename T>
uint64_t all_sym_invertible_count(int32_t dim) {
    uint64_t ans = 0;
    int32_t param_length = (dim * (dim + 1)) / 2;
    int64_t mat_number = 1 << param_length;
    for (int64_t ind = 0; ind < mat_number; ++ind) {
        arma::Mat<T> m(dim, dim);
        int32_t digit = 0;
        for (int32_t row = 0; row < dim; ++row) {
            for (int32_t col = row; col < dim; ++col) {
                m(row, col) = m(col, row) = T((ind >> digit) & 1);
                ++digit;
            }
        }
        if (arma::rank(arma::conv_to<arma::mat>::from(m)) == dim) {
            ++ans;
        }
    }
    return ans;
}

template<typename T>
void test_all_sym_invertible_speed(std::ostream &out = std::cout) {
    auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    for (int32_t d = 1; d < 8; ++d) {
        out << "Number of sym inv for " << d << " is " << all_sym_invertible_count<int>(d) << "\n";
        now = std::chrono::steady_clock::now();
        out << "Time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << '\n';
        start = now;
    }
}

int main()
{
    test_all_sym_invertible_speed<int>();
    return 0;
}
