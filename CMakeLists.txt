cmake_minimum_required(VERSION 3.12)
project(diploma)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER g++-8)

#set(CMAKE_CXX_FLAGS "-fsanitize=address -lpthread ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-fsanitize=thread -lpthread ${CMAKE_CXX_FLAGS}")


add_executable(diploma src/main.cpp src/utils.cpp src/neuron.cpp src/neural_network.cpp)

add_executable(test src/test.cpp src/utils.cpp src/neuron.cpp src/neural_network.cpp)
