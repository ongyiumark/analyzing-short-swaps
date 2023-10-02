#include "headers/generator.h"
#include "headers/converter.h"

#include <iostream>

InsertGenerator::InsertGenerator(int _N, int _M, std::string DIR) 
  : Generator(_N, _M, (DIR.size() ? DIR : "insert-"+std::to_string(_M))) {}

std::vector<long long> InsertGenerator::get_allowed_moves(long long u) {
  std::vector<long long> moves;
  std::vector<int> p = get_permutation_from_index(u, N);
  for (int i = 0; i < N; i++) {
    for (int j = 1; j < M; j++) {
      if (i+j < N) {
        for (int k = i; k < i+j; k++) std::swap(p[k], p[k+1]);
        moves.emplace_back(get_index_of_permutation(p));
        for (int k = i+j-1; k >= i; k--) std::swap(p[k], p[k+1]);
      }
      if (i-j >= 0) {
        for (int k = i; k > i-j; k--) std::swap(p[k], p[k-1]);
        moves.emplace_back(get_index_of_permutation(p));
        for (int k = i-j+1; k <= i; k++) std::swap(p[k], p[k-1]);
      }
    }
  }
  return moves;
}

int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    std::cerr << "Expected 2 or 3 parameters, but received " << argc-1 << "." << std::endl;  
    return 0;
  } 

  int N = atoi(argv[1]), M = atoi(argv[2]);
  std::string DIR = (argc == 4 ? argv[3] : "");
  InsertGenerator g(N, M, DIR);
  g.generate();
  return 0;
}