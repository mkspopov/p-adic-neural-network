#include <algorithm>
#include <fstream>

#include "neural_network.h"


template
class NeuralNetwork<int(int, int, int)>;

template
double Accuracy(const NeuralNetwork<int(int, int, int)>& network, const DataSet& data);

template
void SetWeightsAndTest(NeuralNetwork<int(int, int, int)>& network, Weights& weights, double& max_acc,
                       const DataSet& dataset, int input_dim);

int DefaultActivationFunction(int value, int num_digits, int prime) {
    int indicator = (value % PrimeToK(prime, num_digits) == 0);
    return 1 - indicator;
}

template<class AFt>
NeuralNetwork<AFt>::NeuralNetwork(int input_dim, int output_dim, const std::vector<int>& dims,
                                  int num_digits, int prime)
        : input_dim_(input_dim), output_dim_(output_dim), prime_(prime), num_digits_(num_digits),
          prime_to_k_(PrimeToK(prime, num_digits)), zero_(num_digits_, 0) {

    assert(!dims.empty());
    assert(output_dim == dims.back());
    layers_.emplace_back(ConstructLayer(dims[0], input_dim));
    for (size_t i = 1; i < dims.size(); ++i) {
        layers_.emplace_back(ConstructLayer(dims[i], dims[i - 1]));
    }
}

template<class AFt>
using Layer = std::vector<Neuron<AFt>>;
using Domain = std::unordered_set<Point>;
using Prediction = int;

template<class AFt>
Layer<AFt>
NeuralNetwork<AFt>::ConstructLayer(int cur_dim, int prev_dim, const std::vector<Weights>& weights) {
    Layer layer(cur_dim);
    if (weights.empty()) {
        for (auto& neuron : layer) {
            neuron.SetZeroWeights(prev_dim);
        }
    } else {
        for (size_t i = 0; i < layer.size(); ++i) {
            layer[i].SetWeights(weights[i]);
        }
    }
    return layer;
}

// for prime == 2
template<class AFt>
Domain
NeuralNetwork<AFt>::GetLearningDomainApproximation(const DataElem& data_elem, int dim) {
    /// epsilon == prime to the power of {-num_zeros}
    Domain domain;
    auto center = GetSatisfyingPoint(data_elem);
    std::vector<const Point*> prev_points;
    for (int num_zeros = num_digits_ - 1; num_zeros >= 0; --num_zeros) {
        prev_points.clear();
        for (int i = 0; i < num_iterations_; ++i) {
            auto it = domain.emplace(GetSatisfyingPointOnSphere(data_elem, center, num_zeros, dim));
            if (it.second) {
                prev_points.push_back(&*it.first);
            }
        }
        if (prev_points.empty()) {
            return domain;
        }
        center = *GetRandomFromVector(prev_points);
    }
    return domain;
}

// for prime == 2
template<class AFt>
void NeuralNetwork<AFt>::GetLearningDomainApproximation(const DataSet& dataset, int dim, int ind) {
    /// epsilon == prime to the power of {-num_zeros}
    auto center = GetSatisfyingPoint(dataset[ind]);
    std::vector<const Point*> prev_points;
    for (int num_zeros = num_digits_ - 1; num_zeros >= 0; --num_zeros) {
        prev_points.clear();
        for (int i = 0; i < num_iterations_; ++i) {
            auto it = domains_[ind].emplace(GetSatisfyingPointOnSphere(dataset[ind], center,
                                                                       num_zeros, dim));
            if (it.second) {
                prev_points.push_back(&*it.first);
                domains_vectorized_[ind].push_back(*it.first);
            }
        }
        if (prev_points.empty()) {
            return;
        }
        center = *GetRandomFromVector(prev_points);
    }
}

template<class AFt>
Point NeuralNetwork<AFt>::GetSatisfyingPoint(const DataElem& data_elem) const {
    auto point = GetVectorFromNkN();
    int cur_iteration = 0;
    while (cur_iteration < max_iterations_ && !IsSatisfied(data_elem, point)) {
        point = GetVectorFromNkN();
        ++cur_iteration;
    }
    return point;
}

template<class AFt>
bool NeuralNetwork<AFt>::Learn(const DataSet& dataset) {
    std::vector<Domain> approx_domains(dataset.size());
//    for (const auto& pair : dataset) {
//        approx_domains.emplace_back(GetLearningDomainApproximation(.));
//    }
    return true;
}

template<class AFt>
inline void NeuralNetwork<AFt>::SetLearningTime(int num_iterations, int num_possible_minimums,
                                                int max_iterations) {
    num_iterations_ = num_iterations;
    num_possible_minimums_ = num_possible_minimums;
    num_possible_minimums_initial_ = num_possible_minimums_;
    max_iterations_ = max_iterations;
}

template<class AFt>
std::vector<Point> NeuralNetwork<AFt>::Predict(const DataSet& dataset) const {
    std::vector<Point> predictions;
    predictions.reserve(dataset.size());
    for (const auto& pair : dataset) {
        predictions.push_back(Predict(pair));
    }
    return predictions;
}

template<class AFt>
Point NeuralNetwork<AFt>::Predict(const DataElem& data_pair) const {
    Point values = data_pair.first;
    Point new_values;

    new_values.reserve(values.size());
    for (size_t l = 0; l < layers_.size(); ++l) {
        new_values.clear();
        for (auto& neuron : layers_[l]) {
            new_values.push_back(neuron(values, num_digits_, prime_));
        }
        std::swap(values, new_values);
    }
    return values;
}

template<class AFt>
void NeuralNetwork<AFt>::SetRandomWeights() {
    for (size_t l = 0; l < layers_.size(); ++l) {
        for (auto& neuron : layers_[l]) {
            neuron.SetWeights({}, num_digits_, prime_);
        }
    }
}

template<class AFt>
void NeuralNetwork<AFt>::SetActivationFunction(AFt* func, int layer) {
    if (layer == -1) {
        for (size_t i = 0; i < layers_.size(); ++i) {
            SetActivationFunction(func, i);
        }
    } else {
        for (auto& neuron : layers_[layer]) {
            neuron.SetActivationFunction(func);
        }
    }
}

template<class AFt>
void NeuralNetwork<AFt>::Init(AFt* func, int num_iterations, int num_possible_minimums,
                              int max_iterations) {
    SetActivationFunction(func);
    SetLearningTime(num_iterations, num_possible_minimums, max_iterations);
}

template<class AFt>
bool NeuralNetwork<AFt>::LearnOneLayerBetter(const DataSet& dataset, bool skip_right_answer,
                                             bool is_next_epoch) {
    if (!is_next_epoch) {
        domains_.clear();
        domains_vectorized_.clear();
        num_possible_minimums_ = num_possible_minimums_initial_;
    }
    for (auto& neuron : layers_.back()) {
        size_t ind = 0;
        for (const auto& data_elem : dataset) {
//            auto answer = Predict(data_elem);
//            if (skip_right_answer && answer[0] == data_elem.second) {
//                continue;
//            }
            if (ind >= domains_.size()) {
                domains_.push_back(GetLearningDomainApproximation(data_elem, input_dim_));
                domains_vectorized_.emplace_back();
                for (const auto& point : domains_.back()) {
                    domains_vectorized_.back().push_back(point);
                }
            } else {
                GetLearningDomainApproximation(dataset, input_dim_, ind);
            }
            ++ind;
        }
        /// got the vector of approx domains

        /// now we need to find a minimum among these domains
        if (domains_.empty()) {
            return true;
        }
        auto min_weights = FindMinimumOneLayer(dataset);
//        for (auto& vec : domains_vectorized_) {
//            vec.clear();
//        }
        neuron.SetWeights(min_weights);
    }
    num_possible_minimums_ += num_possible_minimums_initial_;
    return true;
}

std::ofstream max_count_file("output/max_count.txt");

template<class AFt>
Point NeuralNetwork<AFt>::FindMinimumOneLayer(const DataSet& dataset) {
    int max_count = 0;
    Point best_weights;
    for (const auto& domain : domains_vectorized_) {
        for (int i = 0; i < num_possible_minimums_; ++i) {
            auto ind = GetRandomNumber(0, domain.size() - 1);
            auto count = CountPoint(dataset, domain[ind]);
            if (count > max_count) {
                max_count = count;
                best_weights = domain[ind];
            }
        }
//        std::cerr << max_count << '\n';
    }
    // TODO: not only for one output neuron
//    std::cerr << max_count << " --- max_count\n";
    if (max_count > max_count_) {
        max_count_ = max_count;
        best_weights_ = best_weights;
    }
    max_count_file << max_count_ << ' ';
    return best_weights_;
}

template<class AFt>
inline Point NeuralNetwork<AFt>::GetRandomPointFromDomain(const Domain& domain) const {
    assert(!domain.empty());
    return GetRandomFromUnorderedSet(domain);
}

template<class AFt>
Point NeuralNetwork<AFt>::FindMinimumOneLayer(const std::vector<Domain>& domains) const {
    std::vector<Point> points(num_possible_minimums_);
    std::vector<int> counts(num_possible_minimums_, 0);
    int max_count = INT32_MIN;
    int max_index = INT32_MIN;
    assert(!domains.empty());
    for (int i = 0; i < num_possible_minimums_; ++i) {
        auto ind = GetRandomNumber(0, static_cast<int>(domains.size()) - 1);
        auto point = GetRandomPointFromDomain(domains[ind]);
        points[i] = std::move(point);

        /// go through all domains
        for (const auto& domain : domains) {
            if (IsPointInDomain(points[i], domain)) {
                ++counts[i];
                if (counts[i] > max_count) {
                    max_count = counts[i];
                    max_index = i;
                }
            }
        }
    }
    // max_element is a point contained in most domains
//    std::cerr << max_count << " --- max_count\n";
//    max_count_file << max_count << ' ';
    return points[max_index];
}

template<class AFt>
inline bool NeuralNetwork<AFt>::IsPointInDomain(const Point& point, const Domain& domain) const {
    return domain.find(point) != domain.end();
}

template<class AFt>
Point NeuralNetwork<AFt>::GetVectorFromNkN() const {
    return ::GetVectorFromNkN(input_dim_, num_digits_, prime_);
}

template<class AFt>
bool NeuralNetwork<AFt>::IsSatisfied(const DataElem& data_elem, const Point& point) const {
    auto dot = DotProd(data_elem.first, point);
    if (data_elem.second == 1) {
        return !IsInBall(dot, num_digits_, 0);
    }
    // remove if prime != 2
    assert(data_elem.second == 0);
    return IsInBall(dot, num_digits_, 0);
}

template<class AFt>
Point NeuralNetwork<AFt>::GetSatisfyingPointOnSphere(const DataElem& data_elem, const Point& center,
                                                     int num_zeros, int dim) const {
    auto point = GetVectorOnSphereNkN(dim, num_digits_, prime_, num_zeros, center);
    int cur_iteration = 0;
    while (cur_iteration < max_iterations_ && !IsSatisfied(data_elem, point)) {
        point = GetVectorOnSphereNkN(dim, num_digits_, prime_, num_zeros, center);
        ++cur_iteration;
    }
    return point;
}

template<class AFt>
bool NeuralNetwork<AFt>::IsInBall(int point, int exp, int center) const {
    return (point - center) % PrimeToK(prime_, exp) == 0;
}

template<class AFt>
bool NeuralNetwork<AFt>::IsInOneBallButNotInTheOther(int point, int exp_close, int exp_far,
                                                     int center) const {
    return IsInBall(point, exp_far, center) && !IsInBall(point, exp_close, center);
}

template<class AFt>
void NeuralNetwork<AFt>::SetWeightsOutputNeuron(const Weights& weights) {
    layers_.back().back().SetWeights(weights);
}

template<class AFt>
int NeuralNetwork<AFt>::CountPoint(const DataSet& dataset, const Point& point) {
    int count = 0;
    for (const auto& data_elem : dataset) {
        if (IsSatisfied(data_elem, point)) {
            ++count;
        }
    }
    return count;
}

template<class AFt>
double Accuracy(const NeuralNetwork<AFt>& network, const DataSet& data) {
    auto answer = network.Predict(data);
    double num_right = 0;
    for (size_t i = 0; i < answer.size(); ++i) {
        if (answer[i][0] == data[i].second) {
            ++num_right;
        }
    }
    return num_right / data.size();
}

template<class AFt>
void SetWeightsAndTest(NeuralNetwork<AFt>& network, Weights& weights, double& max_acc,
                       const DataSet& dataset, int input_dim) {
    if (weights.size() == input_dim) {
        network.SetWeightsOutputNeuron(weights);
        auto acc = std::abs(Accuracy(network, dataset) - 0.5);
        if (acc > 0.3) {
            std::cerr << weights << '\n';
            std::cerr << acc << '\n';
        }
        if (acc > max_acc) {
            max_acc = acc;
            std::cerr << weights << '\n';
            std::cerr << max_acc << " max\n";
        }
        return;
    }
    for (int i = 0; i < 4; ++i) {
        weights.push_back(i);
        SetWeightsAndTest(network, weights, max_acc, dataset, input_dim);
        weights.pop_back();
    }
}
