""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import numpy as np

from util import evaluate, load_data, save_results, compute_accuracy

class PerceptronModel():
    """ Maximum entropy model for classification.

    Attributes:

    """
    def __init__(self, epoch=10, lr=0.01):
        # Initialize the parameters of the model.
        # TODO: Implement initialization of this model.
        self.epoch = epoch
        self.lr = lr

    def train(self, train_data):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        # TODO: Implement the training of this model.
        x_train, y_train = train_data
        self.weight = np.zeros((x_train.shape[1] + 1, y_train.shape[1]))
        x_train = np.concatenate((x_train, np.ones((x_train.shape[0], 1))), axis=1) # add bias to train feature
        for _ in range(self.epoch):
            for i in range(x_train.shape[0]):
                phi = np.dot(x_train[i], self.weight)
                y_pred = np.argmax(phi)
                # print(y_pred)
                if y_train[i][y_pred] != 1:
                    y_true = np.where(y_train[i] == 1)[0][0]
                    self.weight[:, y_pred] -= self.lr * x_train[i]
                    self.weight[:, y_true] += self.lr * x_train[i]
            if (_ % 10 == 0):
                print("Finish training at epoch %d" % _)

    def predict(self, x):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector.

        Returns:
            The predicted class.    

        """
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        return np.argmax(np.dot(x, self.weight), axis=1)

if __name__ == "__main__":
    # Character 2-gram feature vector as default
    # Can be modified in util.py load_data() function
    train_data, dev_data, test_data, data_type, label_map = load_data('propername')
    print('Finish loading data.')

    # Train the model using the training data.
    model = PerceptronModel(epoch=100)
    model.train(train_data)
    print('Finish model training.')

    # Predict on the development set. 
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "perceptron_" + data_type + "_dev_predictions.csv"))
    print('Accuracy in dev set: %f' %dev_accuracy)

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    test_feature, _ = test_data
    y_pred = model.predict(test_feature)
    res = []
    for i in range(len(y_pred)):
        res.append(label_map[y_pred[i]])
    save_results(res, os.path.join("tmp", "perceptron_" + data_type + "_test_predictions.csv"))
    
    # evaluate(model,
    #          test_data,
    #          os.path.join("results", "perceptron_" + data_type + "_test_predictions.csv")) 
