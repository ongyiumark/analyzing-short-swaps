#ifndef CONVERTER_H
#define CONVERTER_H

#include <vector>
#include <iostream>

struct OrderedSet;

std::vector<long long> factorial;

long long get_index_of_permutation(std::vector<int> &p);
std::vector<int> get_permutation_from_index(long long x);

#endif