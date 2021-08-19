import logging
import numpy as np
import progressbar

from scratchkit.utils.misc import bar_widgets
from scratchkit.utils import batch_iterator
from scratchkit.dl.activations import Sigmoid

from sklearn.datasets.openml import fetch_openml
sigmoid = Sigmoid()

class RBM():
    """
    Bernoulli Restricted Boltzmann Machine (RBM)

    Parameters:
    -----------
    n_hidden: int:
        The number of processing nodes (neurons) in the hidden layer. 
    learning_rate: float
        The step length that will be used when updating the weights.
    batch_size: int
        The size of the mini-batch used to calculate each weight update.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    
    """
    def __init__(self, n_hidden=128, learning_rate=0.1, batch_size=10, n_iterations=100):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lr = learning_rate
        self.n_hidden = n_hidden
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def _initialize_weights(self, X):
        n_visible = X.shape[1]
        self.W = np.random.normal(scale=0.1, size=(n_visible, self.n_hidden))
        self.w0 = np.zeros(n_visible) # Bias visible
        self.h0 = np.zeros(self.n_hidden) # Bias hidden

    def fit(self, X, y=None):
        self._initialize_weights(X)

        self.training_errors = []
        self.training_reconstructions = []
        for _ in self.progressbar(range(self.n_iterations)):
            batch_errors = []
            for batch in batch_iterator(X, batch_size=self.batch_size):
                # Positive phase
                positive_hidden = sigmoid(batch.dot(self.W) + self.h0)
                hidden_states = self._sample(positive_hidden)
                positive_associations = batch.T.dot(positive_hidden)

                # Negative phase
                negative_visible = sigmoid(hidden_states.dot(self.W.T) + self.v0)
                negative_visible = self._sample(negative_visible)
                negative_hidden = sigmoid(negative_visible.dot(self.W) + self.h0)
                negative_associations = negative_visible.T.dot(negative_hidden)

                self.W  += self.lr * (positive_associations - negative_associations)
                self.h0 += self.lr * (positive_hidden.sum(axis=0) - negative_hidden.sum(axis=0))
                self.v0 += self.lr * (batch.sum(axis=0) - negative_visible.sum(axis=0))

                batch_errors.append(np.mean((batch - negative_visible) ** 2))

            self.training_errors.append(np.mean(batch_errors))
            # Reconstruct a batch of images from the training set
            idx = np.random.choice(range(X.shape[0]), self.batch_size)
            self.training_reconstructions.append(self.reconstruct(X[idx]))

    def _sample(self, X):
        return X > np.random.random_sample(size=X.shape)

    def reconstruct(self, X):
        positive_hidden = sigmoid(X.dot(self.W) + self.h0)
        hidden_states = self._sample(positive_hidden)
        negative_visible = sigmoid(hidden_states.dot(self.W.T) + self.v0)
        return negative_visible

if __name__ == '__main__':
    mnist = fetch_openml('mnist_784')

    X = mnist.data
    y = mnist.target

    model = RBM()
    model.fit(X, y) 
