#include <vector>
#include <map>
#include <queue>
#include <string>

#include <iostream>
#include <sstream>
#include <fstream>

#include <cassert>
#include <algorithm>
#include <numeric>

typedef std::pair<int,int> ii;
typedef long long ll;
typedef std::vector<int> vi;
typedef std::vector<vi> vvi;

int N, M;
std::string DIR;
constexpr int INF = 1e9;

void print(vi &p) {
  for (int i = 0; i < N; i++) std::cout << p[i] << (i+1 < N ? " " : "\n");
}

void build_mapping(vvi &perms, std::map<vi,int> &idx) {
  vi p(N);

  ll sz = 1;
  for (int i = 1; i <= N; i++) sz *= i;
  perms.resize(sz);

  iota(p.begin(), p.end(), 1);
  int i = 0;
  do {
    idx[p] = i;
    perms[i++] = p;
  } while(next_permutation(p.begin(), p.end()));
}

void build_adj_list(vvi &adj, vvi &perms, std::map<vi,int> &idx) {
  for (auto &pp : perms) {
    int u = idx[pp];
    for (int i = 0; i < N; i++) {
      for (int k = 1; k < M; k++) {
        if (i+k >= N) continue;
  
        for (int j = i; j < i+k; j++) std::swap(pp[j], pp[j+1]);
        int v = idx[pp];

        adj[u].push_back(v);

        for (int j = i+k-1; j >= i; j--) std::swap(pp[j], pp[j+1]);
      }
    }
  }
}

void bfs(int s, vvi &adj, vi &par) {
  std::vector<bool> vis(adj.size());

  std::queue<int> q;
  q.push(s);
  vis[s] = true;

  while(!q.empty()) {
    int u = q.front();
    q.pop();

    for (int &v : adj[u]) {
      if (vis[v]) continue;
    
      q.push(v);
      vis[v] = true;
      par[v] = u;
    }
  }
}

void generate_csv(vvi &perms, vi &par) {
  std::ostringstream s;
  if (DIR.size()) s << "./../data/" << DIR << "/perm" << N << ".csv"; 
  else s << "./../data/" << M << "insert/perm" << N << ".csv";
  std::ofstream file(s.str());

  for (int i = 0; i < N; i++) file << "a" << i+1 << ",";
  for (int i = 0; i < N; i++) file << "b" << i+1 << (i+1 < N ? "," : "\n");

  for (int u = 1; u < perms.size(); u++) {
    int v = par[u];

    for (int i = 0; i < N; i++) file << perms[u][i] << ",";
    for (int i = 0; i < N; i++) file << perms[v][i] << (i+1 < N ? "," : "\n");
  }

  file.close();
}

int main(int argc, char* argv[]) {
  N = atoi(argv[1]);
  M = atoi(argv[2]);
  assert(1 < M && M <= N);

  if (argc > 3) DIR = argv[3];

  vvi perms; // stores all permuations
  std::map<vi, int> idx; // maps a permutation to its index
  build_mapping(perms, idx);

  vvi adj(perms.size());
  build_adj_list(adj, perms, idx); // adjacency list of possible swaps

  vi par(adj.size());
  bfs(0, adj, par); // find the shortest paths using bfs

  generate_csv(perms, par); // generate csv dataset
}