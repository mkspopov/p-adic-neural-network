#pragma once

#include <unordered_map>

#include "neural_network.h"


class PyPlot {
public:
    PyPlot() {
        std::cin >> num_cases_;
    }

    void ReadInput() {

    }

    void Run() {
        for (int i = 0 ; i < num_cases_; ++i) {
            std::getline(std::cin, line_);
            size_t pos = 0;
            size_t next = line_.find('\'', pos);
            while (next != std::string::npos) {
                pos = line_.find('\'', next + 1);
                params_.emplace(line_.substr(next + 1, pos), GetNum(pos));
                next = line_.find('\'', pos + 1);
            }
        }
    }

    int GetNum(size_t pos) {
        while (!std::isdigit(line_[pos])) {
            ++pos;
        }
        int num = 0;
        while (std::isdigit(line_[pos])) {
            num *= 10;
            num += line_[pos] - '0';
            ++pos;
        }
        return num;
    }

private:
    int num_cases_ = 0;
    std::string line_;
    std::unordered_map<std::string, int> params_;
};
