#define ARMA_DONT_USE_STD_MUTEX
#include <iostream>
#include <vector>
#include <chrono>
#include <armadillo>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/isomorphism.hpp>
#include "cube-matrices.h"

const double EPS = 1e-6;

bool close_to_01(double number) {
    return std::abs(number) < EPS || std::abs(number - 1) < EPS;
}

template<typename T>
class SymInvertibleGenerator {
    int32_t dim;
    int64_t max_index;
    int64_t index;
    arma::Mat<T> mat;

public:
    explicit SymInvertibleGenerator(int32_t dim) : dim(dim), index(0) {
        int32_t param_num = (dim * (dim + 1)) / 2;
        max_index = 1 << param_num;
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

arma::Mat<int16_t> filtered_matrix(const arma::mat &prods_matrix) {
    arma::Mat<int16_t> result(prods_matrix.n_rows, prods_matrix.n_cols);
    for (int32_t row = 0; row < result.n_rows; ++row) {
        for (int32_t col = 0; col < result.n_cols; ++col) {
            result(row, col) = (close_to_01(prods_matrix(row, col)) ? int16_t(0) : int16_t(1));
        }
    }
    return result;
}

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> Graph;

Graph column_poset_graph(const arma::mat &b01, const arma::mat &cube) {
    arma::mat prods_matrix = cube.t() * arma::inv(b01) * cube;
    arma::Mat<int16_t> filter = filtered_matrix(prods_matrix);
    Graph graph(filter.n_cols);
    for (int32_t first = 0; first < filter.n_cols; ++first) {
        for (int32_t second = 0; second < filter.n_cols; ++second) {
            if (first == second) continue;
            bool good = true;
            for (int32_t row = 0; row < filter.n_rows; ++row) {
                if (filter(row, first) < filter(row, second)) {
                    good = false;
                    break;
                }
            }
            if (good) {
                boost::add_edge(first, second, graph);
            }
        }
    }
    return graph;
}

std::vector<std::pair<arma::mat, Graph>> all_sym_inv_diff_graphs(int32_t dim) {
    arma::mat cube = cubens::c<double>(dim);
    SymInvertibleGenerator<double> gen(dim);
    std::vector<std::pair<arma::mat, Graph>> dif_graphs_pairs;
    while (gen) {
        arma::mat b01 = gen();
        Graph new_graph = column_poset_graph(b01, cube);
        bool is_new = true;
        for (const auto &pr : dif_graphs_pairs) {
            if (boost::graph::isomorphism(new_graph, pr.second)) {
                is_new = false;
                break;
            }
        }
        if (is_new) {
            dif_graphs_pairs.emplace_back(b01, new_graph);
        }
    }
    return dif_graphs_pairs;
}

void test_sym_dif_graph_speed(std::ostream &out = std::cout) {
    auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    for (int32_t d = 1; d < 7; ++d) {
        out << "Number of different graphs of sym inv for " << d << " is " << all_sym_inv_diff_graphs(d).size() << "\n";
        now = std::chrono::steady_clock::now();
        out << "Time in milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() << '\n';
        start = now;
    }
}

int main()
{
    test_sym_dif_graph_speed();
    //for (const auto &pr: all_sym_inv_diff_graphs(3)) {
    //   std::cout << "Next b01 is:\n" << pr.first << '\n';
    //}
    //arma::mat b01_1({{1, 0, 1}, {0, 1, 0}, {1, 0, 0}});
    //arma::mat b01_2({{1, 1, 1}, {1, 1, 0}, {1, 0, 0}});
    //std::cout << boost::graph::isomorphism(column_poset_graph(b01_2, cubens::c3), column_poset_graph(b01_1, cubens::c3));
    return 0;
}
