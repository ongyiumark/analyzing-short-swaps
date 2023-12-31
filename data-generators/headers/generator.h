#ifndef GENERATOR_H
#define GENERATOR_H

#include <string>
#include <vector>
#include <utility>

void createDir(std::string dir);

class Generator {
protected:
  int N, M;
  std::string DIR;
  std::vector<std::vector<long long>> adj_list;
  std::vector<int> distance;
  std::vector<long long> moves; 
  long long sz;

public:
  Generator(int _N, int _M, std::string _DIR);
  virtual void get_possible_moves();
  std::vector<long long> get_allowed_moves(long long u);
  void build_adj_list();
  void bfs(int s);
  virtual void save_to_csv();
  void generate();
  int get_distance(int u);
};

class SwapGenerator : public Generator {
public:
  SwapGenerator(int _N, int _M, std::string DIR);
  void get_possible_moves() override;
  void save_to_csv() override;
  std::pair<int,int> get_move_from_state(int u, int v);
};

class InsertGenerator : public Generator {
public:
  InsertGenerator(int _N, int _M, std::string DIR);
  void get_possible_moves() override;
};

class ReverseGenerator : public Generator {
public:
  ReverseGenerator(int _N, int _M, std::string DIR);
  void get_possible_moves() override;
  void save_to_csv() override;
  std::pair<int,int> get_move_from_state(int u, int v);
};

class BlockGenerator : public Generator {
public:
  BlockGenerator(int _N, int _M, std::string DIR);
  void get_possible_moves() override;
};

#endif