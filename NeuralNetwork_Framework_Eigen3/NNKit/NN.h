#include <Eigen/Eigen>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// Use these type definations incase of changing data types like `float` to `double`
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf colVector;

// Neural Net 
class NN {
    public:
        NN(std::vector<uint32_t> topology, Scalar lr = Scalar(0.005)); // lr -> Learning Rate
        // Forward Function
        void forward(RowVector& input);
        // Backward Function
        void backward(RowVector& output);
        // Calculate Errors for each neuron in each Layer
        void calcError(RowVector& output);
        // Update weights of connections
        void update();
        // Train function
        void train(std::vector<RowVector*> data);

        /**
         * Storage Objects for the working of the NN Class
        */ 
        std::vector<RowVector*> neuronLayers; // Stores the layers of Output Network.
        std::vector<RowVector*> cacheLayers; // Stores the unactivated values of the layers.
        std::vector<RowVector*> deltas; // Stores the error contribution of each neurons.
        std::vector<Matrix*> weights; 
        Scalar lr; // Learning Rate.
};

// Constructor
NN::NN(std::vector<uint32_t> topology, Scalar lr) {
    this->topology = topology;
    this->lr = lr;

    for (uint32_t i = 0; i < topology.size(); i++) {
        // Initialize Neuron Layers
        if (i == topology.size() - 1) {
            neuronLayers.push_back(new RowVector(topology[i]));
        } else {
            neuronLayers.push_back(new RowVector(topology[i] + 1));
        }

        // Initialize Cache and delta vectors
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        // vector.back() gives the handle to recently added element
        // coeffRef gives the reference of value at that place.
        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // Initialize Weight Matrix
        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i-1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i-1], topology[i]) = 1.0;
            } else {
                weights.push_back(new Matrix(topology[i-1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
}

void NN::forward(RowVector& input) {
    // Set Input to Input Layer
    /**
     * block @return a part of the given Vector or Matrix.
     * `block` takes 4 args: startRow, startCol, blockRows, blockCols.
     */
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    // Propogate data forward
    for (uint32_t i = 1; i < topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
    }

    // Apply Activation function to the net.
    // UnaryExpr applies the given function to all elements of CURRENT_LAYER
    for (uint32_t i = 1; i < topology.size() - 1; i++) {
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activation));
    }
}

void NN::update() {
    // topology.size()-1 = weights.size()
    for (uint32_t i = 0; i < topology.size() - 1; i++) {
        // in this loop we are iterating over the different layers (from first hidden to output layer)
        // if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
        // if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
        if (i != topology.size() - 2) {
            for (uint32_t c = 0; c < weights[i]->cols() - 1; c++) {
                for (uint32_t r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += lr * deltas[i + 1]->coeffRef(c) * activationDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else {
            for (uint32_t c = 0; c < weights[i]->cols(); c++) {
                for (uint32_t r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += lr * deltas[i + 1]->coeffRef(c) * activationDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

// Backward
void NN::backward(RowVector& output) {
    calcError(output);
    update(); // Update Weights
}

Scalar activation(Scalar x) {
    return tanhf(x);
}

Scalar activationDerivative(Scalar x) {
    return 1 - tanhf(x) * tanhf(x);
}

// Train Function
void NN::train(std::vector<RowVector*> input, std::vector<RowVector*> output) {
    for (uint32_t i = 0; i < input.size(); i++) {
        std::cout << "Input: " << *input[i] << std::endl;
        forward(*input[i]);
        std::cout << "Expected output is: " << *output[i] << std::endl;
        std::cout << "Output is: " << *neuronLayers.back() << std::endl;
        backward(*output[i]);
        std::cout << "MSE: " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
    }
}

// ReadCSV
void ReadCSV(std::string filename, std::vector<RowVector*>& data) {
    data.clear();
    std::ifstream file(filename);
    std::string line, word;
    // determine number of columns in file
    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<Scalar> parsed_vec;
    while (getline(ss, word, ', ')) {
        parsed_vec.push_back(Scalar(std::stof(&word[0])));
    }
    uint32_t cols = parsed_vec.size();
    data.push_back(new RowVector(cols));
    for (uint32_t i = 0; i < cols; i++) {
        data.back()->coeffRef(1, i) = parsed_vec[i];
    }
 
    // read the file
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            std::stringstream ss(line);
            data.push_back(new RowVector(1, cols));
            uint32_t i = 0;
            while (getline(ss, word, ', ')) {
                data.back()->coeffRef(i) = Scalar(std::stof(&word[0]));
                i++;
            }
        }
    }
}

void genData(std::string filename) {
    std::ofstream file1(filename + "-in");
    std::ofstream file2(filename + "-out");
    for (uint32_t r = 0; r < 1000; r++) {
        Scalar x = rand() / Scalar(RAND_MAX);
        Scalar y = rand() / Scalar(RAND_MAX);
        file1 << x << ", " << y << std::endl;
        file2 << 2 * x + 10 + y << std::endl;
    }
    file1.close();
    file2.close();
}

/**
// main.cpp
 
// don't forget to include out neural network
#include "NeuralNetwork.hpp"
 
//... data generator code here
 
typedef std::vector<RowVector*> data;
int main()
{
    NeuralNetwork n({ 2, 3, 1 });
    data in_dat, out_dat;
    genData("test");
    ReadCSV("test-in", in_dat);
    ReadCSV("test-out", out_dat);
    n.train(in_dat, out_dat);
    return 0;
}
*/