#include "headers/converter.h"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
  const bool to_permuation = (argc >= 2 && strcmp(argv[1], "-i") == 0);
  const bool to_index = (argc >= 2 && strcmp(argv[1], "-p") == 0);

  // Error checking
  if (!to_permuation && !to_index) std::cerr << "Please select a correct option!" << std::endl;
  if (to_permuation && argc != 4) std::cerr << "Expected 2 parameters, but received " << argc-2 << "." << std::endl;  
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