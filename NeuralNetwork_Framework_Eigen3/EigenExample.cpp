/**
 * This File is to Demonstrate How to Run a .cpp with #include <Eigen/> in it
 *
 * I used this command to run this file. First Install and load eigen3 in this dir or any dir then run this.
 *
 * g++ EigenExample.cpp -I /path/to/eigen/eigen-3.4.0 --verbose ...
 *
 */
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

int main() {
    MatrixXd m(10, 10);
    m(0, 0) = 3;
    m(1, 0) = 2.5;
    m(0, 1) = -1;
    m(1, 1) = m(1, 0) + m(0, 1);
    std::cout << m << std::endl;
}