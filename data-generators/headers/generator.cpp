#include "generator.h"
#include "converter.h"

#include <queue>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

#if _MSC_VER
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

void createDir(std::string dir) {
  #if _MSC_VER
  _mkdir(dir.data());
  #elif _WIN32
  mkdir(dir.data());
  #else
  mkdir(dir.data(), 0777);
  #endif
}

Generator::Generator(int _N, int _M, std::string _DIR) : N(_N), M(_M), DIR(_DIR) {
  sz = 1;
  for (int i = 1; i <= N; i++) sz *= i;
  adj_list.resize(sz);
  distance.resize(sz);
}

int Generator::get_distance(int u) {
  return distance[u];
}

void Generator::get_possible_moves() {}

std::vector<long long> Generator::get_allowed_moves(long long u) {
  std::vector<long long> allowed_moves;
  std::vector<int> p = get_permutation_from_index(u, N);
  for (long long &v : moves) {
    std::vector<int> move_p = get_permutation_from_index(v, N);
    std::vector<int> new_p(N);
    for (int i = 0; i < N; i++) new_p[i] = p[move_p[i]-1];

    allowed_moves.push_back(get_index_of_permutation(new_p));
  }
  return allowed_moves;
}

void Generator::build_adj_list() {
  for (long long u = 0; u < sz; u++) {
    if (u % 10000 == 0) std::cerr << "Adjacency list: " << u << "/" << sz << std::endl;
    std::vector<long long> possible_moves = get_allowed_moves(u);
    for (int v : possible_moves) adj_list[u].emplace_back(v);
  }
}

void Generator::bfs(int s) {
  std::vector<bool> visited(sz);

  std::queue<int> q;
  q.push(s);
  visited[s] = true;
  distance[s] = 0;

  while(!q.empty()) {
    int u = q.front();
    q.pop();

    for (long long &v : adj_list[u]) {
      if (visited[v]) continue;
      q.push(v);
      visited[v] = true;
      distance[v] = distance[u]+1;
    }
  }
}

void Generator::save_to_csv() {
  std::string directory = "./../data/" + DIR;  
  createDir(directory);

  std::ostringstream s;
  s << "./../data/" << DIR << "/perm" << N << ".csv"; 
  std::ofstream file(s.str());
  file << "state,next_state,distance\n";
  for (int u = 1; u < sz; u++) {
    if (u % 100000 == 0) std::cerr << "CSV: " << u << "/" << sz << std::endl;
    std::vector<long long> possible_moves = get_allowed_moves(u);
    std::sort(possible_moves.begin(), possible_moves.end());

    int parent = -1;
    for (long long &v : possible_moves) {
      if (distance[v] != distance[u]-1) continue;
      parent = v; break;
    }

    file << u << "," << parent << "," << distance[u] << "\n";
  }
  
  file.close();
}

void Generator::generate() {
  std::cerr << "Building adjacency list..." << std::endl;
  build_adj_list();

  std::cerr << "Running breadth-first search..." << std::endl;
  bfs(0);

  std::cerr << "Saving to file..." << std::endl;
  save_to_csv();
}

