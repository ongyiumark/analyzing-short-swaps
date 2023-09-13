#include <vector>
#include <iostream>
#include <queue>
#include <map>
#include <bitset>
#include <string>

typedef long long ll;
typedef std::vector<int> vi;
typedef std::vector<vi> vvi;

std::vector<ll> factorial;
constexpr int N = 1e8;
std::bitset<N> vis;

// binary-indexed tree
struct ordered_set {
  vi BIT;
  int sz;

  ordered_set(int n) {
    sz = 1;
    while (sz < n) sz *= 2;
    BIT.resize(sz+1);
  }

  void insert(int x) {
    for (int i = x; i <= sz; i += i&(-i)) BIT[i] += 1;
  }

  void erase(int x) {
    for (int i = x; i <= sz; i += i&(-i)) BIT[i] -= 1;
  }

  int find_by_order(int k) {
    int total = 0;
    int res = 0;
    for (int i = sz; i >= 1; i >>= 1) {
      if (total + BIT[res+i] <= k) {
        total += BIT[res+i];
        res += i;
      }
    }
    return res+1;
  }

  int order_of_key(int x) {
    int total = 0;
    for (int i = x; i > 0; i -= i&(-i)) {
      total += BIT[i];
    }
    return total-1;
  }
};

// get index of a permutation
ll get_idx(vi &p) {
  int n = p.size();

  ordered_set os(n);
  for (int i = 1; i <= n; i++) os.insert(i);
  
  ll ans =  0;
  for (int i = 0; i < n; i++) {
    int j = os.order_of_key(p[i]);
    ans += j*factorial[n-i-1];
    os.erase(p[i]);
  }
  return ans;
}

// get permuation from index
vi get_permuation(ll x, int n) {
  vi p(n);

  ordered_set os(n);
  for (int i = 1; i <= n; i++) os.insert(i);

  for (int i = 0; i < n; i++) {
    int lo = 0;
    int hi = n-i-1;
    int ans = 0;

    while (lo <= hi) {
      int mid = hi - (hi - lo)/2;
      if (mid*factorial[n-i-1] <= x) {
        ans = mid;
        lo = mid+1;
      }
      else hi = mid-1;
    }

    p[i] = os.find_by_order(ans);
    os.erase(p[i]);
    x -= ans*factorial[n-i-1];
  }

  return p;
}

int bfs(ll s, int n) {
  std::queue<ll> q;
  q.push(s);
  vis.set(s);

  int moves = 0;
  int current_layer = 1;
  int next_layer = 0;

  while(!q.empty()) {
    int u = q.front();
    q.pop();
    if (u == 0) return moves;

    vi p = get_permuation(u, n);
    for (int i = 0; i < n; i++) {
      for (int j = 1; j <= 2; j++) {
        if (i + j >= n) continue;
        if (p[i] < p[i+j]) continue; // skip non correcting swaps
        std::swap(p[i], p[i+j]);

        ll v = get_idx(p);
        if (!vis[v]) {
          q.push(v);
          next_layer++;
          vis.set(v);
        }

        std::swap(p[i], p[i+j]);
      }
    }

    current_layer--;
    if (current_layer == 0) {
      current_layer = next_layer;
      next_layer = 0;
      moves++;
    }
  }
  return -1;
}

int main(int argc, char* argv[]) {
  std::string command = argv[1];
  int n = atoi(argv[2]);

  factorial.resize(n);
  factorial[0] = 1;
  for (int i = 1; i < n; i++) factorial[i] = i*factorial[i-1];

  if (command == "get-permutation") {
    vi perm = get_permuation(atoi(argv[3]), n);
    for (int i = 0; i < n; i++) std::cout << perm[i] << (i+1 < n ? " " : "");
    std::cout << std::endl;
    return 0;
  }

  vi p(n);
  for (int i = 0; i < n; i++) p[i] = atoi(argv[3+i]);

  if (command == "get-index") {
    std::cout << get_idx(p) << std::endl;
    return 0;
  }

  ll s = get_idx(p);
  int moves = bfs(s, n);
  std::cout << moves << "\n";
}