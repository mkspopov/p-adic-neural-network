#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include "neural_network.h"

double kTrainSize = 0.75;
double kValidSize = 0.1;

int SquareActFunc(int dot) {
    return dot * dot;
}

//auto ToOneHot(const std::vector<std::pair<Point, Point>>& data, int dim) {
//    DataSet new_data(data.size());
//    for (size_t i = 0; i < data.size(); ++i) {
//        new_data[i].first = data[i].first;
//        assert(data[i].second[0] >= 0 && data[i].second[0] < dim);
//        new_data[i].second = Point(dim, 0);
//        new_data[i].second[data[i].second[0]] = 1;
//    }
//    return new_data;
//}

auto DataSetBoolFunc(const std::string& func = "xor") {
    DataSet data{
            {{0, 0}, {0}},
            {{0, 1}, {1}},
            {{1, 0}, {1}},
            {{1, 1}, {0}}
    };
    if (func == "or") {
        data[3].second = 1;
    } else if (func == "and") {
        data[1].second = 0;
        data[2].second = 0;
        data[3].second = 1;
    } else {
        assert(func == "xor");
    }
    return std::make_pair(data, data);
}

// if max_iterations == 0 then network get random weights and doesn't learn anything
void TestOneLayerBool(int max_iterations = 100, const std::string& bool_func = "xor") {
    int input_dim = 2;
    int output_dim = 1;
    int num_digits = 2;
    int prime = 2;
    std::vector<int> dims{1};
    NeuralNetwork<decltype(DefaultActivationFunction)> network(input_dim, output_dim, dims,
                                                               num_digits, prime);
    network.SetRandomWeights();
    network.Init(DefaultActivationFunction, max_iterations, max_iterations, max_iterations);
    int num_tests = 1000;

    auto[train, test] = DataSetBoolFunc(bool_func);

    std::vector<int> num_rights;
    for (int t = 0; t < num_tests; ++t) {
        int num_epoches = 100;
        for (int i = 0; i < num_epoches; ++i) {
            network.LearnOneLayerBetter(train, false, false);
        }
        auto answer = network.Predict(test);
        int num_right = 0;
        for (size_t i = 0; i < answer.size(); ++i) {
            if (answer[i][0] == test[i].second) {
                ++num_right;
            }
        }
        num_rights.push_back(num_right);
    }
    std::cout << std::accumulate(num_rights.begin(), num_rights.end(), 0) /
                 static_cast<double>(num_rights.size() * 4) << " --- mean accuracy\n";
    std::cout << *std::max_element(num_rights.begin(), num_rights.end()) /
                 static_cast<double>(4) << " --- max accuracy\n";
}

void TestCompareWithRandomChoiceOfWeights() {
    std::vector<std::string> tests{"or", "xor", "and"};
    for (const auto& test : tests) {
        std::cout << test << " alg: ";
        TestOneLayerBool(10, test);
        std::cout << test << " rand: ";
        TestOneLayerBool(0, test);
    }
}

std::mt19937 gen(37);

template<class BoolFunc>
auto GenerateValues(const BoolFunc& func, int dim) {
    std::vector<Point> points;
    points.reserve(1 << dim);
    Point point;
    for (int i = 0; i < 1 << dim; ++i) {
        point.clear();
        int num = i;
        while (num) {
            point.push_back(num & 1);
            num >>= 1;
        }
        int size = point.size();
        for (int j = 0; j < dim - size; ++j) {
            point.push_back(0);
        }
        points.push_back(point);
    }
    std::shuffle(points.begin(), points.end(), gen);
    std::cout << "Table begin\n";
    for (const auto& point : points) {
        std::cout << point << func(point) << '\n';
    }
    std::cout << "Table end\n\n";
    DataSet train;
    DataSet valid;
    DataSet test;
    for (size_t i = 0; i < 0.75 * points.size(); ++i) {
        train.emplace_back(points[i], func(points[i]));
    }
    for (size_t i = kTrainSize * points.size(); i < (kTrainSize + kValidSize) * points.size();
         ++i) {
        valid.emplace_back(points[i], func(points[i]));
    }
    for (size_t i = (kTrainSize + kValidSize) * points.size(); i < points.size(); ++i) {
        test.emplace_back(points[i], func(points[i]));
    }
    return std::array<DataSet, 3>{train, valid, test};
}

auto CreateNetwork(int input_dim, int num_iterations, int num_possible_minimums, int num_digits) {
    const std::vector<int>& dims = {1};
    int prime = 2;
    int output_dim = dims.back();
    NeuralNetwork<decltype(DefaultActivationFunction)> network(input_dim, output_dim, dims,
                                                               num_digits, prime);
    network.SetRandomWeights();
    network.Init(DefaultActivationFunction, num_iterations, num_possible_minimums);
    return network;
}

template<class BoolFunc>
void FiveArgs(int num_epochs, int dim, int num_iterations, int num_possible_minimums,
              int num_digits, const BoolFunc& func, bool skip_right_answer = false) {
    std::ofstream file("output/boolean" + std::to_string(dim) + ".csv");
    auto[train, valid, test] = GenerateValues(func, dim);
    auto network = CreateNetwork(dim, num_iterations, num_possible_minimums, num_digits);
    for (int i = 0; i < num_epochs; ++i) {
        auto start = std::clock();
        network.LearnOneLayerBetter(train, skip_right_answer, i > 0);
        auto end = std::clock();
        auto duration = (end - start) / static_cast<double> (CLOCKS_PER_SEC);
        std::cerr << "Answer for " << duration << " seconds on epoch " << i << "\n";
        auto train_acc = Accuracy(network, train);
        auto valid_acc = Accuracy(network, valid);
        auto test_acc = Accuracy(network, test);
        std::cerr << train_acc << " --- accuracy on train\n";
        std::cerr << valid_acc << " --- accuracy on valid\n";
        std::cerr << test_acc << " --- accuracy on test\n";
        file << i << ',' << train_acc << ',' << valid_acc << ',' << test_acc << ',' << duration
             << '\n';
    }
}
