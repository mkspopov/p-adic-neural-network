#include <chrono>
#include <fstream>
#include <iomanip>

const double kTrainTestSplit = 0.2;
const double kDatasetSplit = 0.15;
const double kTrainSplit = 0.1;

void PlotNumDigits() {
    std::ofstream file("output/num_digits.csv");
    std::cerr << "Num Digits\n";
    int num_epochs = 100;
    Point num_digs{2, 3, 4, 5, 6, 7};  // TODO: bug with one digit
    auto dataset = ReadDatasetHeartDisease();
    auto[train, test] = SplitDataset(dataset, kTrainTestSplit);
    for (int digs : num_digs) {
        file << digs << ',';
        double acc = 0.0;
        auto start = std::clock();
        TestHeartDisease(num_epochs, train, test, acc, digs);
        auto end = std::clock();
        auto duration = (end - start) / static_cast<double> (CLOCKS_PER_SEC);
        std::cerr << "best_acc : " << acc << '\n';
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "Answer for " << duration << " seconds\n";
        file << acc << ',';
        file << duration << '\n';
    }
}


void PlotNumIterations() {
    std::ofstream file("output/num_iterations.csv");
    std::cerr << "Num Iterations\n";
    int num_tests = 1;
    int num_epochs = 10;
    int num_minimums = 10000;
    Point num_iters;
    int mult = 100;
    for (int i = 1; i <= 20; ++i) {
        num_iters.push_back(i * mult);
    }
    auto dataset = ReadDatasetHeartDisease();
    auto[train, test] = SplitDataset(dataset, kTrainTestSplit);
    for (int iters : num_iters) {
        file << iters << ',';
        double acc = 0.0;
        auto start = std::clock();
        TestHeartDisease(num_epochs, train, test, acc, 2, iters, num_minimums, false, {1}, 2,
                num_tests);
        auto end = std::clock();
        auto duration = (end - start) / static_cast<double> (CLOCKS_PER_SEC);
        std::cerr << "best_acc : " << acc << '\n';
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "Answer for " << duration << " seconds\n";
        file << acc << ',';
        file << duration << '\n';
    }
}

void PlotNumMinimums() {
    std::ofstream file("output/num_minimums.csv");
    std::cerr << "Num Minimums\n";
    Point num_digs{1, 2, 3, 4, 5, 6, 7, 8};
    auto dataset = ReadDatasetHeartDisease();
    auto[train, test] = SplitDataset(dataset, kTrainTestSplit);
    int num_epochs = 100;
    int i = 0;
    for (int digs : num_digs) {
        ++i;
        file << i << ',';
//        file << digs << ',';
        double acc = 0.0;
        auto start = std::clock();
        TestHeartDisease(num_epochs, train, test, acc, 2, 2, digs);
        auto end = std::clock();
        auto duration = (end - start) / static_cast<double> (CLOCKS_PER_SEC);
        std::cerr << "best_acc : " << acc << '\n';
        std::cerr << std::fixed << std::setprecision(3);
        std::cerr << "Answer for " << duration << " seconds\n";
        file << acc << ',';
        file << duration << '\n';
    }
}

void PlotTestHeartDiseaseMy(int num_epochs,
                      int num_digits = 2,
                      int num_iterations = 2,
                      int num_possible_minimums = 2,
                      bool skip_right_answer = false,
                      const std::vector<int>& dims = {1},
                      int prime = 2,
                      int num_tests = 1) {
    std::ofstream file("output/num_epochs.csv");
    int input_dim = 13;
    int output_dim = dims.back();
    NeuralNetwork<decltype(DefaultActivationFunction)> network(input_dim, output_dim, dims,
                                                               num_digits, prime);
    network.SetRandomWeights();
    network.Init(DefaultActivationFunction, num_iterations, num_possible_minimums);
    auto dataset = ReadDatasetHeartDisease();
//    auto train = dataset;
//    auto test = dataset;
    auto[temp_train, test] = SplitDataset(dataset, kDatasetSplit);
    auto[train, valid] = SplitDataset(temp_train, kTrainSplit);
    std::vector<int> num_rights;
    for (int t = 0; t < num_tests; ++t) {
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
//        auto answer = network.Predict(test);
//        int num_right = 0;
//        for (size_t i = 0; i < answer.size(); ++i) {
//            if (answer[i][0] == test[i].second) {
//                ++num_right;
//            }
//        }
//        num_rights.push_back(num_right);
        /// LOG:
//        std::cerr << num_right / static_cast<double>(test.size()) << '\n';
//        std::cerr << Accuracy(network, train) << " --- accuracy on train\n";
//        std::cerr << Accuracy(network, test) << " --- accuracy on test\n";
    }
}
