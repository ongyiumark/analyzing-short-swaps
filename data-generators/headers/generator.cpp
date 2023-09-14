#include "generator.h"

#include <queue>
#include <sstream>
#include <fstream>
#include <iostream>

#if _MSC_VER
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

void createDir(std::string dir) {
  #if _WIN32
  _mkdir(dir.data());
  #else
  mkdir(dir.data(), 0777);
  #endif
}

Generator::Generator(int _N, int _M, std::string _DIR) : N(_N), M(_M), DIR(_DIR) {
  sz = 1;
  for (int i = 1; i <= N; i++) sz *= i;
  adj_list.resize(sz);
  parent.resize(sz);
  distance.resize(sz);
}

std::vector<long long> Generator::get_allowed_moves(long long u) {
  return std::vector<long long>();
}

void Generator::build_adj_list() {
  for (long long u = 0; u < sz; u++) {
    if (u % 100000 == 0) std::cerr << "Adjacency list: " << u << "/" << sz << std::endl;
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
      parent[v] = u;
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
    file << u << "," << parent[u] << "," << distance[u] << "\n";
  }
  
  file.close();
}

void Generator::generate() {
  std::cerr << "Building adjacency list..." << std::endl;
  build_adj_list();

  std::cerr << "Running breadth first search..." << std::endl;
  bfs(0);

  std::cerr << "Saving to file..." << std::endl;
  save_to_csv();
}

