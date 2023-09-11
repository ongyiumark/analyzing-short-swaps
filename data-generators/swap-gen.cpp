#include <bits/stdc++.h>

using namespace std;
typedef pair<int,int> ii;
typedef long long ll;

int N, M;
string DIR;
constexpr int INF = 1e9;

void print(vector<int> &p) {
  for (int i = 0; i < N; i++) cout << p[i] << (i+1 < N ? " " : "\n");
}

void build_mapping(vector<vector<int>> &perms, map<vector<int>,int> &idx) {
  vector<int> p(N);

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

void build_adj_list(vector<vector<int>> &adj, vector<vector<int>> &perms, map<vector<int>,int> &idx) {
  for (auto &pp : perms) {
    int u = idx[pp];
    for (int i = 0; i < N; i++) {
      for (int k = 1; k < M; k++) {
        if (i+k >= N) continue;
        if (pp[i] < pp[i+k]) continue; // ignore noncorrecting swaps (Heath and Vergara, 2003)
  
        swap(pp[i], pp[i+k]);
        int v = idx[pp];

        adj[v].push_back(u); // we will bfs starting from the identity permutation
        swap(pp[i], pp[i+k]);
      }
    }
  }
}

void bfs(int s, vector<vector<int>> &adj, vector<int> &par) {
  vector<bool> vis(adj.size());

  queue<int> q;
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

void generate_csv(vector<vector<int>> &perms, vector<int> &par) {
  ostringstream s;
  if (DIR.size()) s << "./../data/" << DIR << "/perm" << N << ".csv"; 
  else s << "./../data/" << M << "swap/perm" << N << ".csv";
  ofstream file(s.str());

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

  vector<vector<int>> perms; // stores all permuations
  map<vector<int>, int> idx; // maps a permutation to its index
  build_mapping(perms, idx);

  vector<vector<int>> adj(perms.size());
  build_adj_list(adj, perms, idx); // adjacency list of possible swaps

  vector<int> par(adj.size());
  bfs(0, adj, par); // find the shortest paths using bfs

  generate_csv(perms, par); // generate csv dataset
}