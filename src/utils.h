#pragma once

#include <cassert>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using Point = std::vector<int>;
using DataElem = std::pair<Point, int>;
//using DataElem = std::pair<Point, Point>;
using DataSet = std::vector<DataElem>;

int64_t GetRandomNumber(int left, int right);

//int64_t GetRandomNumberArray(int left, int right);

int64_t GetRandomPAdicNumber(int num_digits, int prime);

// Returns vector from \mathbb{N} _ num_digits ^ dim
std::vector<int> GetVectorFromNkN(int dim, int num_digits, int prime);

// Returns a vector from \mathbb{N} _{num_digits} with num_zeros zeros
auto GetNumberFromNkWithZeros(int num_digits, int prime, int num_zeros, bool on_sphere = true);

std::vector<int> GetVectorOnSphereNkN(int dim, int num_digits, int prime, int num_zeros,
        const std::vector<int>& center);

int PrimeToK(int prime, int k);

int DotProd(const std::vector<int>& values, const std::vector<int>& weights);

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec);

namespace std {
    template <>
    struct hash<std::vector<int>> {
        size_t operator()(const vector<int>& vec) const;
    };
}

template <class Value>
const Value& GetRandomFromUnorderedSet(const std::unordered_set<Value>& set);

template <class Value>
Value GetRandomFromVector(const std::vector<Value>& vec);

template <class Iterable>
void Shuffle(Iterable& train);

std::pair<DataSet, DataSet> SplitDataset(const DataSet& dataset, double for_test);
