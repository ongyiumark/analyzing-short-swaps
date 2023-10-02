#ifndef GENERATOR_H
#define GENERATOR_H

#include <string>
#include <vector>

class Generator {
protected:
  int N, M;
  std::string DIR;
  std::vector<std::vector<long long>> adj_list;
  std::vector<long long> parent;
  std::vector<int> distance;
  long long sz;

public:
  Generator(int _N, int _M, std::string _DIR);
  virtual std::vector<long long> get_allowed_moves(long long u);
  void build_adj_list();
  void bfs(int s);
  void save_to_csv();
  void generate();
};

class SwapGenerator : public Generator {
public:
  SwapGenerator(int _N, int _M, std::string DIR);
  std::vector<long long> get_allowed_moves(long long u) override;
};

class InsertGenerator : public Generator {
public:
  InsertGenerator(int _N, int _M, std::string DIR);
  std::vector<long long> get_allowed_moves(long long u) override;
};

class ReverseGenerator : public Generator {
public:
  ReverseGenerator(int _N, int _M, std::string DIR);
  std::vector<long long> get_allowed_moves(long long u) override;
};

class BlockGenerator : public Generator {
public:
  BlockGenerator(int _N, int _M, std::string DIR);
  std::vector<long long> get_allowed_moves(long long u) override;
};

#endif