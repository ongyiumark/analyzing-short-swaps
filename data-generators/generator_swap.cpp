#include "headers/generator.h"
#include "headers/converter.h"

#include <iostream>
#include <cassert>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

SwapGenerator::SwapGenerator(int _N, int _M, std::string DIR) 
  : Generator(_N, _M, (DIR.size() ? DIR : "swap-"+std::to_string(_M))) {
    get_possible_moves();
  }

void SwapGenerator::get_possible_moves() {
  std::vector<int> p = get_permutation_from_index(0, N);
  for (int i = 0; i < N; i++) {
    for (int j = 1; j < M; j++) {
      if (i+j >= N) continue;
      std::swap(p[i], p[i+j]);
      moves.emplace_back(get_index_of_permutation(p));
      std::swap(p[i], p[i+j]);
    }
  }
}

std::pair<int,int> SwapGenerator::get_move_from_state(int u, int v) {
  std::vector<int> pu = get_permutation_from_index(u, N);
  std::vector<int> pv = get_permutation_from_index(v, N);

  int n = pu.size();
  int i = 0, j = n-1;
  while(pu[i] == pv[i]) i++;
  while(pu[j] == pv[j]) j--;

  std::swap(pu[i], pu[j]); // make move and check
  assert(get_index_of_permutation(pu) == v);

  return std::make_pair(i, j);
}

void SwapGenerator::save_to_csv() {
  std::string directory = "./../data/" + DIR;  
  createDir(directory);

  std::ostringstream s;
  s << "./../data/" << DIR << "/perm" << N << ".csv"; 
  std::ofstream file(s.str());
  file << "state,move\n";
  for (int u = 1; u < sz; u++) {
    if (u % 100000 == 0) std::cerr << "CSV: " << u << "/" << sz << std::endl;
    std::vector<long long> possible_moves = get_allowed_moves(u);
    std::sort(possible_moves.begin(), possible_moves.end());

    int parent = -1;
    for (long long &v : possible_moves) {
      if (distance[v] != distance[u]-1) continue;
      parent = v; break;
    }
    auto [i, j] = get_move_from_state(u, parent);
    file << u << "," << i << "-" << j << "\n";
  }
  
  file.close();
}

#ifndef HAS_MAIN
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
#endif