#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int SZ = 26;
int SmallSZ = 14; // the maximal product ab is 192, so we assume a < 14
int ONES = (1 << SZ) - 1;
std::vector<int> max_pair(SmallSZ, 0); // we assume a < 14

void process_matrix(std::vector<int> mat) {
    for (int msk = 0; msk < (1 << SZ); ++msk) {
        int a = __builtin_popcount(msk);
        if (a >= SmallSZ) { // we assume a < 14
            continue;
        }
        int ban = 0;
        for (int index = 0; index < SZ; ++index) {
            if (msk & (1 << index)) {
                ban |= mat[index];
            }
        }
        int b = SZ - __builtin_popcount(ban);
        if (b > max_pair[a]) {
            max_pair[a] = b;
        }
    }
}

void read_and_process(std::string filename) {
    std::vector<int> mat(SZ,0);
    std::ifstream input_bitcode(filename);
    while (input_bitcode.peek() != EOF) {
        for (int ind = 0; ind < SZ; ++ind) {
            input_bitcode >> mat[ind];
        }
        process_matrix(mat);
    }
}

void write_result(std::string filename) {
    std::ofstream output_file(filename);
    for (int ind = 0; ind < SmallSZ; ++ind) {
        output_file << max_pair[ind] << " ";
    }
    output_file.close();
}

int main() {
    read_and_process("bitcoded5-identity.txt");
    write_result("results-identity.txt");
    return 0;
}
