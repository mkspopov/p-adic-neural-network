#pragma once

#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "neuron.h"


using Point = std::vector<int>;
using DataElem = std::pair<Point, int>;
//using DataElem = std::pair<Point, Point>;
using DataSet = std::vector<DataElem>;

int DefaultActivationFunction(int value, int num_digits, int prime);

template<class AFt>
class NeuralNetwork {
public:
    using Layer = std::vector<Neuron<AFt>>;
    using Domain = std::unordered_set<Point>;

//    NeuralNetwork() = default;

    NeuralNetwork(int input_dim, int output_dim, const std::vector<int>& dims,
                  int num_digits, int prime);

    void Init(AFt* func, int num_iterations, int num_possible_minimums, int max_iterations = 10);

    Layer ConstructLayer(int cur_dim, int prev_dim, const std::vector<Weights>& weights = {});

    void SetRandomWeights();

    void SetWeightsOutputNeuron(const Weights& weights);

    // sets to all layers if layer == -1
    void SetActivationFunction(AFt* func, int layer = -1);

    // TODO: for prime != 2
    Domain GetLearningDomainApproximation(const DataElem& data_elem, int dim);

    void GetLearningDomainApproximation(const DataSet& dataset, int dim, int ind);

    bool Learn(const DataSet& dataset);

    // TODO: for prime != 2
    bool LearnOneLayer(const DataSet& dataset, bool skip_right_answer = false);

    // TODO: for prime != 2
    bool LearnOneLayerBetter(const DataSet& dataset, bool skip_right_answer = false,
                             bool is_next_epoch = false);

    std::vector<Point> Predict(const DataSet& dataset) const;

    Point Predict(const DataElem& data_pair) const;

    void SetLearningTime(int num_iterations, int num_possible_minimums, int max_iterations);

private:
    int input_dim_ = 0;
    int output_dim_ = 0;
    std::vector<Layer> layers_;
    std::vector<Domain> domains_;
    std::vector<std::vector<Point>> domains_vectorized_;
    Point best_weights_;
    int max_count_ = 0;

    int prime_ = 2;
    int num_digits_ = 0;
    int prime_to_k_ = 1;
    Point zero_;

    int num_possible_minimums_ = 10;
    int num_possible_minimums_initial_ = 10;
    int num_iterations_ = 10;
    int max_iterations_ = 10;

    Point FindMinimumOneLayer(const std::vector<Domain>& domains) const;

    Point FindMinimumOneLayer(const DataSet& dataset);

    int CountPoint(const DataSet& dataset, const Point& point);

    Point GetRandomPointFromDomain(const Domain& domain) const;

    bool IsPointInDomain(const Point& point, const Domain& domain) const;

    Point GetVectorFromNkN() const;

    // TODO: for prime != 2
    bool IsSatisfied(const DataElem& data_elem, const Point& point) const;

    Point GetSatisfyingPoint(const DataElem& data_elem) const;

    Point GetSatisfyingPointOnSphere(const DataElem& data_elem, const Point& center,
                                     int num_zeros, int dim) const;

    bool IsInBall(int point, int exp, int center = 0) const;

    bool IsInOneBallButNotInTheOther(int point, int exp_close, int exp_far, int center = 0) const;
};

template<class AFt>
double Accuracy(const NeuralNetwork<AFt>& network, const DataSet& data);

template<class AFt>
void SetWeightsAndTest(NeuralNetwork<AFt>& network, Weights& weights, double& max_acc,
                       const DataSet& dataset, int input_dim);