#define HAS_MAIN
#include "headers/generator.h"
#include "headers/converter.h"
#include "generator_swap.cpp"
#include "generator_insert.cpp"
#include "generator_reverse.cpp"
#include "generator_block.cpp"
#undef HAS_MAIN

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <algorithm>

int main(int argc, char* argv[]) {
  std::string strategy = "";
  int M = -1;
  int N = 8;
  int u = -1;

  bool no_bound = false;
  for (int i = 2; i <= 8; i+=2) {
    if (argc <= i) continue;
    if (strcmp(argv[i-1], "-s") == 0) strategy = argv[i];
    if (strcmp(argv[i-1], "-b") == 0) {
      if (strcmp(argv[i], "n") == 0) no_bound = true;
      else M = atoi(argv[i]);
    }
    if (strcmp(argv[i-1], "-n") == 0) N = atoi(argv[i]);
    if (strcmp(argv[i-1], "-i") == 0) u = atoi(argv[i]);
  }
  if (no_bound) M = N;
  
  std::vector<std::string> valid_strategies = {"swap", "insert", "reserve", "block"};
  if (std::find(valid_strategies.begin(), valid_strategies.end(), strategy) == valid_strategies.end()) {
    if (strategy.size() == 0) std::cerr << "You must specify a strategy with '-s'." << std::endl;
    else std::cerr << "'" << strategy << "'" << " is not a valid strategy." << std::endl;
    return 0;
  }

  if (N <= 0) {
    std::cerr << N << " is not a valid permutation size." << std::endl;
    return 0;
  }

  if (M <= 1 || M > N) {
    if (M == -1) std::cerr << "You must specify a bound with '-b'." << std::endl;
    else std::cerr << M << " is not a valid bound. It must be at least 2 and at most " << N << "." << std::endl;
    return 0;
  }

  long long sz = 1;
  for (long long i = 1; i <= N; i++) sz *= i;
  if (u < 0 || u >= sz) {
    if (u == -1) std::cerr << "You must specify an index with '-i'." << std::endl;
    else std::cerr << u << " is not a valid index. It must be at least 0 and at most " << sz << "." << std::endl;
    return 0;
  }

  Generator *g = nullptr;
  if (strategy == "swap") g = new SwapGenerator(N, M, strategy+"-test");
  else if (strategy == "insert") g = new InsertGenerator(N, M, strategy+"-test");
  else if (strategy == "reverse") g = new ReverseGenerator(N, M, strategy+"-test");
  else if (strategy == "block") g = new BlockGenerator(N, M, strategy+"-test");
  
  std::vector<long long> moves = g->get_allowed_moves(u);
  std::vector<bool> valid_moves(sz);
  for (long long &v : moves) valid_moves[v] = true;

  std::string directory = "./../data/tmp";  
  createDir(directory);

  std::ostringstream s;
  s << "./../data/tmp/" << strategy << "-" << M << "-perm" << N << ".csv"; 
  std::ofstream file(s.str());
  file << "move,valid\n";
  for (int u = 0; u < sz; u++) {
    if (u % 10000 == 0) std::cerr << "CSV: " << u << "/" << sz << std::endl;
    file << u << "," << (valid_moves[u] ? 1 : 0) << "\n";
  }
  
  file.close();
}