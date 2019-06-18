#pragma once

#include <iostream>
#include "utils.h"

using Weights = std::vector<int>;

template<class AFt>
class Neuron {
public:
    Neuron() = default;

    explicit Neuron(const Weights& weights);

    void SetZeroWeights(int size);

    // sets random weights if arg `weights` is empty
    void SetWeights(const Weights& weights = {}, int num_digits = 1, int prime = 2);

    void SetActivationFunction(AFt* func);

    int operator()(const std::vector<int>& values, int num_digits = 1, int prime = 2);

    int operator()(const std::vector<int>& values, int num_digits = 1, int prime = 2) const;

    int operator()(const std::vector<int>& values, int num_digits, int prime, std::string& bot,
                       std::string& mid, std::string& top, int& count);

private:
    Weights weights_;
    AFt* activation_function_ = nullptr;
    int value_ = 0;
    int activ_value_ = 0;
};

void PrintStrings(const std::string& bot, const std::string& mid, const std::string& top);
