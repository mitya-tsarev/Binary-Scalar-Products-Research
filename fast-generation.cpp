#define ARMA_DONT_USE_STD_MUTEX
#include <iostream>
#include <vector>
#include <chrono>
#include <armadillo>
//#include <boost/graph/adjacency_list.hpp>
//#include "cube-matrices.h"

template<typename T>
class SymInvertibleGenerator {
    int32_t dim;
    int64_t max_index;
    int64_t index;
    arma::Mat<T> mat;

public:
    explicit SymInvertibleGenerator(int32_t dim) : dim(dim), index(0) {
        int32_t param_num = (dim * (dim + 1)) / 2;
        max_index = 1 << (((dim + 1) * dim) / 2);
        mat = arma::Mat<T>(dim, dim, arma::fill::zeros);
        (*this)();
    }

    arma::Mat<T> operator()() {
        arma::Mat<T> current_mat = mat;
        do {
            int32_t digit = 0;
            for (int32_t row = 0; row < dim; ++row) {
                for (int32_t col = row; col < dim; ++col) {
                    mat(row, col) = mat(col, row) = T((index >> digit) & 1);
                    ++digit;
                }
            }
            ++index;
        } while (index < max_index && arma::rank(arma::conv_to<arma::mat>::from(mat)) < dim);
        return current_mat;
    }

    explicit operator bool() {
        if (dim > 1) {
            return index < max_index;
        }
        return index < 3; // All-1 matrix is invertible iff dim is 1
    }
};


template<typename T>
uint64_t all_sym_invertible_count(int32_t dim) {
    uint64_t ans = 0;
    SymInvertibleGenerator<T> gen(dim);
    while (gen) {
        ++ans;
        gen();
    }
    return ans;
}

template<typename T>
void test_all_sym_invertible_speed(std::ostream &out = std::cout) {
    auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    for (int32_t d = 1; d < 6; ++d) {
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
