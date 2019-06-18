#include <algorithm>
#include <cassert>

#include "utils.h"


//static const int s_k_SEED = 7;
static const int s_k_SEED = 37;
static std::mt19937 s_gen(s_k_SEED);

int64_t GetRandomNumber(int left, int right) {
    std::uniform_int_distribution<int64_t> dis(left, right);
    return dis(s_gen);
}

template const std::vector<int>& GetRandomFromUnorderedSet(
        const std::unordered_set<std::vector<int>>& set);

template const std::vector<int>* GetRandomFromVector(
        const std::vector<const std::vector<int>*>& set);

template std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<Point>& vec);
template std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec);

template void Shuffle(DataSet& train);

int64_t GetRandomPAdicNumber(int num_digits, int prime) {
    auto p_k = PrimeToK(prime, num_digits);
    std::uniform_int_distribution<int64_t> dis(0, p_k - 1);
    return dis(s_gen);
}

// Returns vector from \mathbb{N} _ num_digits ^ dim
std::vector<int> GetVectorFromNkN(int dim, int num_digits, int prime) {
    std::vector<int> numbers(dim);
    for (auto& num : numbers) {
        num = GetRandomPAdicNumber(num_digits, prime);
    }
    return numbers;
}

auto GetNumberFromNkWithZeros(int num_digits, int prime, int num_zeros, bool on_sphere) {
    std::uniform_int_distribution<int> dis(0, prime - 1);
    int power = PrimeToK(prime, num_zeros);
    int dig = dis(s_gen);
    if (on_sphere) {
        while (dig == 0) {
            dig = dis(s_gen);
        }
    }
    power *= dig + prime * GetRandomPAdicNumber(num_digits - num_zeros - 1, prime);
    return power;
}

std::vector<int> GetVectorOnSphereNkN(int dim, int num_digits, int prime, int num_zeros,
                                      const std::vector<int>& center) {
    assert(dim > 0);
    int prime_to_k = PrimeToK(prime, num_digits);
    std::uniform_int_distribution<int> dis_coord(0, dim - 1);
    std::vector<int> numbers(dim);
    int coord_pos = dis_coord(s_gen);
    for (int i = 0; i < dim; ++i) {
        if (i == coord_pos) {
            numbers[i] = GetNumberFromNkWithZeros(num_digits, prime, num_zeros) + center[i];
            numbers[i] %= prime_to_k;
        } else {
            numbers[i] = GetNumberFromNkWithZeros(num_digits, prime, num_zeros, false) + center[i];
            numbers[i] %= prime_to_k;
        }
    }
    return numbers;
}

int PrimeToK(int prime, int k) {
    if (prime == 2) {
        return 1 << k;
    }
    int answer = 1;
    for (int i = 0; i < k; ++i) {
        answer *= prime;
    }
    return answer;
}

int DotProd(const std::vector<int>& values, const std::vector<int>& weights) {
    int prod = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        prod += values[i] * weights[i];
    }
    return prod;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    for (const auto& el : vec) {
        std::cout << el << ' ';
    }
//    std::cout << '\n';
    return os;
}

size_t std::hash<std::vector<int>>::operator()(const vector<int>& vec) const {
    auto key = vec.size();
    for (auto& i : vec) {
        key ^= i + 0x9e3779b9 + (key << 6) + (key >> 2);
    }
    return key;
}

template <class Value>
inline const Value& GetRandomFromUnorderedSet(const std::unordered_set<Value>& set) {
    return *set.begin();
}

template <class Value>
inline Value GetRandomFromVector(const std::vector<Value>& vec) {
    assert(!vec.empty());
    return vec[GetRandomNumber(0, vec.size() - 1)];
}

template <class Iterable>
void Shuffle(Iterable& train) {
    std::shuffle(train.begin(), train.end(), s_gen);
}

std::pair<DataSet, DataSet> SplitDataset(const DataSet& dataset, double for_test) {
    DataSet train = dataset;
    DataSet test;
    Shuffle(train);
    size_t test_size = dataset.size() * for_test;
    assert(dataset.size() > test_size);
    for (size_t i = 1; i <= test_size; ++i) {
        test.push_back(train[train.size() - i]);
    }
    train.resize(dataset.size() - test_size);
    return std::make_pair(train, test);
}
