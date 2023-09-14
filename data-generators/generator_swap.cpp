#include "headers/generator.h"
#include "headers/converter.h"

#include <iostream>

struct SwapGenerator : Generator {
  SwapGenerator(int _N, int _M, std::string DIR) 
    : Generator(_N, _M, (DIR.size() ? DIR : "swap-"+std::to_string(_M))) {}

  std::vector<long long> get_allowed_moves(long long u) override {
    std::vector<long long> moves;
    std::vector<int> p = get_permutation_from_index(u, N);
    for (int i = 0; i < N; i++) {
      for (int j = 1; j < M; j++) {
        if (i+j >= N) continue;
        std::swap(p[i], p[i+j]);
        moves.emplace_back(get_index_of_permutation(p));
        std::swap(p[i], p[i+j]);
      }
    }
    return moves;
  }
};

int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    std::cerr << "Expected 2 or 3 parameters, but received " << argc-1 << "." << std::endl;  
    return 0;
  } 

  int N = atoi(argv[1]), M = atoi(argv[2]);
  std::string DIR = (argc == 4 ? argv[3] : "");
  SwapGenerator g(N, M, DIR);
  g.generate();
  return 0;
}