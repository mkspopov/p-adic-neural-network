#pragma once

#include <bitset>
#include <cassert>
#include <iostream>
#include <unordered_set>

#include "utils.h"


void TestGenerators() {
    int num_tests = 100;
    int dim = 100;
    int prime = 2;
    int num_digits = 9;
    int num_zeros = 5;
    std::vector<int> center(dim, 0);
    for (int t = 0; t < num_tests; ++t) {
        auto numbers = GetVectorOnSphereNkN(dim, num_digits, prime, num_zeros, center);
        bool found = false;
        for (auto& number : numbers) {
//            std::cout << std::bitset<10>(number) << '\n';
            assert(number < static_cast<int>(std::pow(prime, num_digits)));
            assert(number % static_cast<int>(std::pow(prime, num_zeros)) == 0);
            if (number % static_cast<int>(std::pow(prime, num_zeros + 1)) != 0) {
                found = true;
            }
        }
        assert(found);
    }
    for (int t = 0; t < num_tests; ++t) {
        auto numbers = GetVectorFromNkN(dim, num_digits, prime);
        for (auto& number : numbers) {
//            std::cout << std::bitset<10>(number) << '\n';
            assert(number < 1 << num_digits);
        }
    }
}

void TestUnorderedSetOfVectors() {
    std::unordered_set<std::vector<int>> set;
    std::vector<int> vec{1, 2, 10};
    set.emplace(vec);
    set.emplace(3, 2);
    assert(set.find({1, 2}) == set.end());
    assert(set.find({2, 2, 2}) != set.end());
    assert(set.find({1, 2, 10}) != set.end());

    // delete if GetRandomFromUnorderedSet has been changed
    assert(*set.begin() == GetRandomFromUnorderedSet(set));
}
