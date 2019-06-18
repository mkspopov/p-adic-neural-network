#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include "neural_network.h"


auto ReadDatasetCancer() {
    std::ifstream file("../datasets/breast-cancer-wisconsin.data");
    assert(file.is_open());
    int number = -1;
    char comma = ',';
    DataSet dataset;
    DataElem data;
    while (file >> number) {
        data.first.clear();
        assert(number >= 0);
        file >> comma;
        for (int i = 0; i < 9; ++i) {
            file >> number;
            assert(number >= 0);
            file >> comma;
            data.first.push_back(number);
        }
        file >> number;
        assert(number >= 0);
        data.second = number == 2 ? 0 : 1;
        dataset.push_back(data);
    }
    assert(dataset.size() == 683);
    return dataset;
}

void TestCancerOneThread() {
    std::ofstream file("output/cancer.csv");
    int num_epochs = 50;
    int num_digits = 2;
    int num_iterations = 1;
    int num_possible_minimums = 1;
    int prime = 2;
    bool skip_right_answer = false;
    const std::vector<int>& dims = {1};

    auto dataset = ReadDatasetCancer();
    const double kTestSize = 0.15;
    const double kValidSize = 0.1;
    auto[train_temp, test] = SplitDataset(dataset, kTestSize);
    auto[train, valid] = SplitDataset(dataset, kValidSize);
//    auto train = dataset;
//    auto test = dataset;

    int input_dim = 9;
    int output_dim = dims.back();
    NeuralNetwork<decltype(DefaultActivationFunction)> network(input_dim, output_dim, dims,
                                                               num_digits, prime);
    network.SetRandomWeights();
    network.Init(DefaultActivationFunction, num_iterations, num_possible_minimums);

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

void TestCancerBruteForce(int input_dim,
                          int num_digits = 2,
                          int num_iterations = 0,
                          int num_possible_minimums = 1,
                          int prime = 2,
                          bool skip_right_answer = true,
                          const std::vector<int>& dims = {1}) {
    int output_dim = dims.back();
    auto dataset = ReadDatasetCancer();
    NeuralNetwork<decltype(DefaultActivationFunction)> network(input_dim, output_dim, dims,
                                                               num_digits, prime);
    network.Init(DefaultActivationFunction, num_iterations, num_possible_minimums);
    Weights weights;
    double max_acc = 0;
    SetWeightsAndTest(network, weights, max_acc, dataset, input_dim);
    std::cout << (max_acc < 0.5 ? 0.5 + max_acc : max_acc) << " --- max acc\n";
}
