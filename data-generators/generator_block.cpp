#include "headers/generator.h"
#include "headers/converter.h"

#include <iostream>

BlockGenerator::BlockGenerator(int _N, int _M, std::string DIR) 
  : Generator(_N, _M, (DIR.size() ? DIR : "block-"+std::to_string(_M))) {}

std::vector<long long> BlockGenerator::get_allowed_moves(long long u) {
  std::vector<long long> moves;
  std::vector<int> p = get_permutation_from_index(u, N);
  
  for (int i = 0; i < N; i++) {
    for (int j = 1; j < M; j++) {
      if (i+j >= N) continue;
      // block moves from i to i+j inclusive
      for (int k = i; k < i+j; k++) {
        int left_sz = k-i+1;
        int right_sz = i+j-k;

        std::vector<int> next_p(N);
        for (int idx = 0; idx < N; idx++) {
          if (idx < i) next_p[idx] = p[idx];
          else if (idx < i+right_sz) next_p[idx] = p[i+left_sz + (idx-i)];
          else if (idx < i+right_sz+left_sz) next_p[idx] = p[i+(idx-i-right_sz)];
          else next_p[idx] = p[idx];
        }
        moves.emplace_back(get_index_of_permutation(next_p));
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
  BlockGenerator g(N, M, DIR);
  g.generate();
  return 0;
}