#include "headers/generator.h"
#include "headers/converter.h"

#include <iostream>
#include <algorithm>

ReverseGenerator::ReverseGenerator(int _N, int _M, std::string DIR) 
  : Generator(_N, _M, (DIR.size() ? DIR : "reverse-"+std::to_string(_M))) {
    get_possible_moves();
  }

void ReverseGenerator::get_possible_moves() {
  std::vector<int> p = get_permutation_from_index(0, N);
  for (int i = 0; i < N; i++) {
    for (int j = 1; j < M; j++) {
      if (i+j >= N) continue;
      std::reverse(p.begin()+i, p.begin()+i+j+1);
      moves.emplace_back(get_index_of_permutation(p));
      std::reverse(p.begin()+i, p.begin()+i+j+1);
    }
  }
}

#ifndef HAS_MAIN
int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    std::cerr << "Expected 2 or 3 parameters, but received " << argc-1 << "." << std::endl;  
    return 0;
  } 

  int N = atoi(argv[1]), M = atoi(argv[2]);
  std::string DIR = (argc == 4 ? argv[3] : "");
  ReverseGenerator g(N, M, DIR);
  g.generate();
  return 0;
}
#endif