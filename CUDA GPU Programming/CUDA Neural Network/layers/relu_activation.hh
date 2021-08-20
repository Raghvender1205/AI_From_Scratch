#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
    private:
        Matrix A;

        Matrix Z;
        Matrix dZ;

    public:
        ReLUActivation(std::string name);
        ~ReLUActivation();

        Matrix& forward(Matrix& Z);
        Matrix& backward(Matrix& dA, float learning_rate = 0.01);
};
