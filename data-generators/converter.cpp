#include "converter.h"

// Ordered Statistics Tree
struct OrderedSet {
  std::vector<int> BIT;
  int sz;

  OrderedSet(int n) {
    sz = 1;
    while (sz < n) {
      sz <<= 1;
    }
    BIT.resize(sz+1, 0);
  }

  void insert(int x) {
    while (x <= sz) {
      BIT[x] += 1;
      x += (x&-x);
    }
  }

  void erase(int x) {
    while (x <= sz) {
      BIT[x] -= 1;
      x += (x&-x);
    }
  }

  int order_of_key(int x) {
    int order = 0;
    while (x > 0) {
      order += BIT[x];
      x -= (x&-x);
    }
    return order-1;
  }

  int find_by_order(int k) {
    int order = 0;
    int result = 0;
    for (int i = sz; i > 0; i >>= 1) {
      if (order + BIT[i+result] <= k) {
        order += BIT[i+result];
        result += i;
      }
    }

    return result+1;
  }
};

std::vector<int> get_permutation_from_index(long long k, int n) {
  std::vector<int> p(n);
  OrderedSet ordered_set(n);
  for (int i = 0; i < n; i++) ordered_set.insert(i+1);

  if (factorial.size() < n) {
    factorial.resize(n);
    factorial[0] = 1;
    for (int i = 1; i < n; i++) factorial[i] = i*factorial[i-1];
  }

  for (int i = 0; i < n; i++) {
    int lo = 0, hi = n-i-1, ans = 0;
    while (lo <= hi) {
      int mid = hi - (hi-lo)/2;
      if (mid*factorial[n-i-1] <= k) {
        ans = mid;
        lo = mid+1;
      }
      else hi = mid-1;
    }

    p[i] = ordered_set.find_by_order(ans);
    k -= ans*factorial[n-i-1];
    ordered_set.erase(p[i]);
  }
  return p;
}

long long get_index_of_permutation(std::vector<int> &p) {
  int n = p.size();
  OrderedSet ordered_set(n);
  for (int i = 0; i < n; i++) ordered_set.insert(i+1);

  if (factorial.size() < n) {
    factorial.resize(n);
    factorial[0] = 1;
    for (int i = 1; i < n; i++) factorial[i] = i*factorial[i-1];
  }

  long long index = 0;
  for (int i = 0; i < n; i++) {
    index += ordered_set.order_of_key(p[i])*factorial[n-i-1];
    ordered_set.erase(p[i]);
  }
  return index;
}

int main(int argc, char* argv[]) {
  const bool to_permuation = (argc >= 2 && strcmp(argv[1], "-i") == 0);
  const bool to_index = (argc >= 2 && strcmp(argv[1], "-p") == 0);

  // Error checking
  if (!to_permuation && !to_index) std::cerr << "Please select a correct option!" << std::endl;
  if (to_permuation && argc != 4) std::cerr << "Expected 2 parameters, but received " << argc-2 << "." << std::endl;;  
  if (to_index && argc < 3) std::cerr << "Expected at least 1 parameter, but received 0." << std::endl;

  if (to_permuation && argc == 4) {
    std::vector<int> p = get_permutation_from_index(atoi(argv[2]), atoi(argv[3]));
    int n = p.size();
    for (int i = 0; i < n; i++) std::cout << p[i] << (i+1 < n ? " " : "\n");
    std::cout << std::flush;
  }

  if (to_index && argc >= 3) {
    int n = argc-2;
    std::vector<int> p(n);
    for (int i = 0; i < n; i++) p[i] = atoi(argv[2+i]);
    std::cout << get_index_of_permutation(p) << std::endl;
  }

  return 0;
}