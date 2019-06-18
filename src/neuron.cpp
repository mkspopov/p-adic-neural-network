#include <cassert>

#include "neuron.h"

template class Neuron<int (int, int, int)>;

template<class AFt>
Neuron<AFt>::Neuron(const Weights& weights) : weights_(weights) {
}

template<class AFt>
void Neuron<AFt>::SetZeroWeights(int size) {
    for (int i = 0; i < size; ++i) {
        weights_.push_back(0);
    }
}

// sets random weights if arg `weights` is empty
template<class AFt>
void Neuron<AFt>::SetWeights(const Weights& weights, int num_digits, int prime) {
    if (!weights.empty()) {
        weights_ = weights;
        return;
    }
    for (auto& weight : weights_) {
        weight = static_cast<int>(GetRandomPAdicNumber(num_digits, prime));
    }
}

template<class AFt>
int Neuron<AFt>::operator()(const std::vector<int>& values, int num_digits, int prime) {
    value_ = DotProd(values, weights_);
    activ_value_ = activation_function_(value_, num_digits, prime);
    return activ_value_;
}

template<class AFt>
int Neuron<AFt>::operator()(const std::vector<int>& values, int num_digits, int prime) const {
    auto value = DotProd(values, weights_);
    return activation_function_(value, num_digits, prime);
}

template<class AFt>
void Neuron<AFt>::SetActivationFunction(AFt* func) {
    activation_function_ = func;
}

template<class AFt>
int Neuron<AFt>::operator()(const std::vector<int>& values, int num_digits, int prime, std::string& bot,
                            std::string& mid, std::string& top, int& count) {
    value_ = DotProd(values, weights_);
    activ_value_ = activation_function_(value_, num_digits, prime);
    if (count % 3 == 1) {
        bot += std::to_string(value_);
        bot += ", ";
        bot += std::to_string(activ_value_);
        bot += "      ";
    } else if (count % 3 == 0) {
        top += std::to_string(value_);
        top += ", ";
        top += std::to_string(activ_value_);
        top += "      ";
    } else {
        mid += "      " + std::to_string(value_) + ", " + std::to_string(activ_value_);
    }
    ++count;
    return activ_value_;
}

void PrintStrings(const std::string& bot, const std::string& mid, const std::string& top) {
    std::cout << top << '\n';
    std::cout << '\n';
    std::cout << mid << '\n';
    std::cout << '\n';
    std::cout << bot << '\n';
}
