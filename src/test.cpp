#include "test_heart_diseases.h"
#include "test_cancer_diseases.h"
#include "test_plot.h"
#include "test_boolean.h"
#include "test_utils.h"
#include "utils.h"

void TestUtils() {
    TestUnorderedSetOfVectors();
    TestGenerators();
}

void SmallNNTestsCout() {
//    TestOneLayerBool();
//    std::string bot, mid, top;
//    PrintStrings(bot, mid, top);
//    TestXSquared();
    TestCompareWithRandomChoiceOfWeights();
}

void Plot() {
//    PlotNumDigits();
//    PlotNumIterations();
//    PlotNumMinimums();
    PlotTestHeartDiseaseMy(50, 2, 10, 10);
}

void TestBoolean() {
    auto func5 = [](const Point& x) -> bool {
        return (x[1] && x[2]) || (x[3] && x[4] && x[0]);
    };
    auto func8 = [](const Point& x) -> bool {
        return (x[1] && x[2]) || (x[3] && x[4] && x[0]) || (x[1] &&
        x[5] ^x[6]) ||(x[1] && x[7]);
    };
    auto func10 = [](const Point& x) -> bool {
        return (x[1] && x[2]) || (x[3] && x[4] && x[0]) ^
                                         x[5] || (x[6] && (x[7] ^ x[8] ^ x[9]));
    };
    auto func12 = [](const Point& x) -> bool {
        return (x[1] && x[2]) || (x[3] && x[4] && x[0]) ^ x[5] || ((x[6] && (x[7] ^ x[8] ^ x[9]))
        && x[10] ^ x[11]);
    };
    int num_epochs = 10;
    int num_minimums = 10;
    int num_iterations = 10;
    int num_digits = 2;
    FiveArgs(num_epochs, 5, num_iterations, num_minimums, num_digits, func5);
    FiveArgs(num_epochs, 8, num_iterations, num_minimums, num_digits, func8);
    FiveArgs(num_epochs, 10, num_iterations, num_minimums, num_digits, func10);
    FiveArgs(num_epochs, 12, num_iterations, num_minimums, num_digits, func12);
}

void TestCancerDataset() {
//    TestCancerOneThread();
    TestCancerBruteForce(9, 2, 0, 10, 2, true, {1});
}

void TestHeartDiseasesDataset() {
    std::cout << TestHeartDiseaseOneThread(50, 2, 1, 1) << '\n';
//    TestHeartDiseasesBruteForce();
}

int main() {
//    TestUtils();
//    SmallNNTestsCout();
//    Plot();
//    TestBoolean();
//    TestHeartDiseasesDataset();
//    TestCancerDataset();
}
