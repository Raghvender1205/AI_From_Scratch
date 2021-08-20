#pragma once
#include "nn_layer.hh"

/// For unit testing purposes
namespace {
    class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	class LinearLayerTest_ShouldReturnDerivativeAfterBackward_Test;
	class LinearLayerTest_ShouldUptadeItsBiasDuringBackward_Test;
	class LinearLayerTest_ShouldUptadeItsWeightsDuringBackward_Test;
}

class LinearLayer : public NNLayer {
    private:
        const float weights_init_threshold = 0.01;

        Matrix W;
        Matrix b;

        Matrix Z;
        Matrix A;
        Matrix dA;

        void initializeBiasWithZeros();
        void initializeWeightsRandomly();

        void computeAndStoreBackwardError(Matrix& dZ);
        void computeAndStoreLayerOutput(Matrix& A);
        void updateWeights(Matrix& dZ, float learning_rate);
        void updateBias(Matrix& dZ, float learning_rate);

    public:
        LinearLayer(std::string name, Shape W_shape);
        ~LinearLayer();

        Matrix& forward(Matrix& A);
        Matrix& backward(Matrix& dZ, float learning_rate = 0.01);

        int getXDim() const;
        int getYDim() const;

        Matrix getWeightsMatrix() const;
        Matrix getBiasMatrix() const;

        /// For Unit Testing
        friend class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
        friend class NeuralNetworkTest_ShouldPerformForwardProp_Test;
        friend class LinearLayerTest_ShouldReturnDerivativeAfterBackward_Test;
        friend class LinearLayerTest_ShouldUptadeItsBiasDuringBackward_Test;
        friend class LinearLayerTest_ShouldUptadeItsWeightsDuringBackward_Test;
};
