""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as F

from util import evaluate, load_data

class MaximumEntropyModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self, alpha=0):
        # Initialize the parameters of the model.
        # TODO: Implement initialization of this model.
        self.weight = None
        self.xdim = None
        self.ydim = None
        self.N = None
        self.alpha = alpha


    def train(self, training_data):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        # TODO: Implement training of this model.
        x_train, y_train = training_data
        self.N = x_train.shape[0]
        self.xdim = x_train.shape[1]
        self.ydim = y_train.shape[1]
        # fmin_l_bfgs_b only support 1d array
        self.weight = np.zeros(self.xdim * self.ydim)
        self.weight, min_val, info = F(func=self.objective, 
                                        x0=self.weight,
                                        fprime=self.calc_grad,
                                        args=(x_train, y_train),
                                        maxiter=150
                                        )

    def calc_prob(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def objective(self, *args):
        w, x_train, y_train = args
        y_train_ = np.where(y_train == 1)[1]
        w = w.reshape(self.xdim, self.ydim)
        tmp = np.dot(x_train, w)
        prod = self.calc_prob(tmp)
        L = 0
        for i in range(self.N):
            L += np.log(prod[i][np.where(y_train[i] == 1)[0][0]])
        L -= 0.5 * self.alpha * np.sum(w ** 2)
        print('Loss: %f' %L)
        return -L

    def calc_grad(self, *args):
        w, x_train, y_train = args
        w = w.reshape(self.xdim, self.ydim)
        w_prime = np.zeros((self.xdim, self.ydim))
        tmp = np.dot(x_train, w)
        prod = self.calc_prob(tmp)
        for i in range(self.ydim):
            tmp = np.zeros((1, self.xdim))
            for j in range(self.N):
                if y_train[j][i] == 1:
                    tmp += x_train[j][:]
            w_prime[:, i] += tmp[0]
            w_prime[:, i] -= np.sum(x_train * prod[:, i].reshape(-1, 1), axis=0)
        w_prime -= self.alpha * w

        w_prime /= self.N
        return -w_prime.flatten()


    def predict(self, model_input):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector.

        Returns:
            The predicted class.    

        """
        # TODO: Implement prediction for an input.
        w = self.weight.reshape(self.xdim, self.ydim)
        return np.argmax(np.dot(model_input, w), axis=1)

if __name__ == "__main__":
    train_data, dev_data, test_data, data_type = load_data(sys.argv)

    # Train the model using the training data.
    model = MaximumEntropyModel()
    model.train(train_data)

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "maxent_" + data_type + "_dev_predictions.csv"))

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", "maxent_" + data_type + "_test_predictions.csv"))
