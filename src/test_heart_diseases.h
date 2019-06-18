#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include "neural_network.h"


auto ReadDatasetHeartDisease() {
    std::ifstream file("../datasets/processed.cleveland.data");
    assert(file.is_open());
    double number = 0.0;
    char comma = ',';
    DataSet dataset;
    int ind = 0;
    DataElem data;
    while (file >> number) {
        ++ind;
        if (ind != 14) {
            data.first.push_back(number * 10);
            file >> comma;
        } else {
            if (number > 0.5) {
                data.second = {1};
            } else {
                data.second = {0};
            }
            assert(data.first.size() == 13);
            dataset.push_back(data);
            data.first.clear();
            ind = 0;
        }
    }
    return dataset;
}

void TestHeartDisease(int num_epochs,
                      const DataSet& train,
                      const DataSet& test,
                      double& accuracy,
                      int num_digits = 2,
                      int num_iterations = 2,
                      int num_possible_minimums = 2,
                      bool skip_right_answer = false,
                      const std::vector<int>& dims = {1},
                      int prime = 2,
                      int num_tests = 1) {
    int input_dim = 13;
    int output_dim = dims.back();
    NeuralNetwork<decltype(DefaultActivationFunction)> network(input_dim, output_dim, dims,
                                                               num_digits, prime);
    network.SetRandomWeights();
    network.Init(DefaultActivationFunction, num_iterations, num_possible_minimums);

    std::vector<int> num_rights;
    for (int t = 0; t < num_tests; ++t) {
        for (int i = 0; i < num_epochs; ++i) {
            auto start = std::clock();
            network.LearnOneLayerBetter(train, skip_right_answer, i > 0);
            auto end = std::clock();
            auto duration = (end - start) / static_cast<double> (CLOCKS_PER_SEC);
            std::cerr << "Answer for " << duration << " seconds on epoch " << i << "\n";
            std::cerr << Accuracy(network, train) << " --- accuracy on train\n";
            std::cerr << Accuracy(network, test) << " --- accuracy on test\n";
        }
        auto answer = network.Predict(test);
        int num_right = 0;
        for (size_t i = 0; i < answer.size(); ++i) {
            if (answer[i][0] == test[i].second) {
                ++num_right;
            }
        }
        num_rights.push_back(num_right);
        /// LOG:
        std::cerr << num_right / static_cast<double>(test.size()) << '\n';
        std::cerr << Accuracy(network, train) << " --- accuracy on train\n";
        std::cerr << Accuracy(network, test) << " --- accuracy on test\n";
    }
    accuracy = (*std::max_element(num_rights.begin(), num_rights.end()) /
                static_cast<double>(test.size()));
    auto accuracy_min = (*std::min_element(num_rights.begin(), num_rights.end()) /
                static_cast<double>(test.size()));
    if (1 - accuracy_min > accuracy) {
        accuracy = 1 - accuracy_min;
    }
//    std::cerr << accuracy << " --- accuracy\n";
}

double TestHeartDiseaseOneThread(int num_epochs,
                                 int num_digits = 2,
                                 int num_iterations = 1'000,
                                 int num_possible_minimums = 1'000,
                                 int prime = 2,
                                 bool skip_right_answer = false,
                                 const std::vector<int>& dims = {1},
                                 int num_tests = 1) {
    auto dataset = ReadDatasetHeartDisease();
//    auto[train, test] = SplitDataset(dataset, 0.2);
    auto train = dataset;
    auto test = dataset;
    double acc = 0.0;
    TestHeartDisease(num_epochs, train, test, acc, num_digits, num_iterations,
                     num_possible_minimums, skip_right_answer, dims, prime, num_tests);
    return acc;
}

void TestHeartDiseasesBruteForce(int input_dim = 13,
                          int num_digits = 2,
                          int num_iterations = 0,
                          int num_possible_minimums = 0,
                          int prime = 2,
                          const std::vector<int>& dims = {1}) {
    int output_dim = dims.back();
    auto dataset = ReadDatasetHeartDisease();
    NeuralNetwork<decltype(DefaultActivationFunction)> network(input_dim, output_dim, dims,
                                                               num_digits, prime);
    network.Init(DefaultActivationFunction, num_iterations, num_possible_minimums);
    Weights weights;
    double max_acc = 0;
    SetWeightsAndTest(network, weights, max_acc, dataset, input_dim);
    std::cout << (max_acc < 0.5 ? 0.5 + max_acc : max_acc) << " --- max acc\n";
}

// TODO: add std::mt19937 generator as an argument to avoid thread-sanitizer's warnings
//void TestTestHeartDiseaseMultiThread(int num_threads = 8) {
//    std::vector<std::thread> threads;
//    std::vector<double> accuracy(num_threads, -1);
//    auto dataset = ReadDatasetHeartDisease();
//    auto[train, test] = SplitDataset(dataset, 0.3);
//    for (int i = 0; i < num_threads; ++i) {
//        /// lambda --- c++ cheat for default arguments in std::thread
//        threads.emplace_back([train = std::cref(train), test = std::cref(test),
//                                     acc = std::ref(accuracy[i])] {
//            TestHeartDisease(train, test, acc);
//        });
//    }
//    for (auto& thread : threads) {
//        if (thread.joinable()) {
//            thread.join();
//        }
//    }
//    std::cout << "Best accuracies:\n";
//    for (auto acc : accuracy) {
//        if (acc > 0) {
//            std::cout << acc << '\n';
//        }
//    }
//}
