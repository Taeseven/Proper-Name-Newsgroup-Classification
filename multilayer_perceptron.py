""" Maximum entropy model for Assignment 1: Starter code.

You can change this code however you like. This is just for inspiration.

"""
import os
import sys
import dynet as dy
import numpy as np
import random
import time

from util import evaluate, load_data

class MultilayerPerceptronModel():
    """ Maximum entropy model for classification.

    Attributes:

    """

    def __init__(self, vocab_size, class_num, hidden1_size=128, hidden2_size=64, final_size=64):
        # Initialize the parameters of the model.
        # TODO: Implement initialization of this model.
        self.dyparams = dy.DynetParams()
        self.dyparams.set_autobatch(True)
        self.dyparams.init()

        self.vocab_size = vocab_size
        self.class_num = class_num
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.final_size = final_size
        self.pc = dy.ParameterCollection()

        self.word_embeddings = self.pc.add_lookup_parameters((self.vocab_size, self.hidden1_size), name="word-embeddings")

        self.hidden1_weights = self.pc.add_parameters((self.hidden2_size, self.hidden1_size), name="hidden1-weights", init=dy.NormalInitializer())
        self.hidden1_biases = self.pc.add_parameters((self.hidden2_size, 1), name="hidden1-biases", init=dy.NormalInitializer())

        self.hidden2_weights = self.pc.add_parameters((self.final_size, self.hidden2_size), name="hidden2-weights", init=dy.NormalInitializer())
        self.hidden2_biases = self.pc.add_parameters((self.final_size, 1), name="hidden2-biases", init=dy.NormalInitializer())

        self.final_weights = self.pc.add_parameters((self.final_size, self.class_num), name = "final-weights")
        self.final_biases = self.pc.add_parameters((self.class_num, 1), name = "final-biases")

        self.optimizer = dy.AdamTrainer(self.pc)

    def get_score(self, feature):
        word_vectors = []
        for i in range(len(feature)):
            if feature[i] == 1:
                word_vectors.append(self.word_embeddings[i])

        embedding = dy.zeros((self.hidden1_size, 1))
        if len(word_vectors) != 0:
            embedding = dy.esum(word_vectors) / float(len(word_vectors))

        intermediate_value = self.hidden1_weights * dy.reshape(embedding, (self.hidden1_size, 1)) + self.hidden1_biases
        intermediate_value = dy.tanh(intermediate_value)

        intermediate_value = self.hidden2_weights * intermediate_value + self.hidden2_biases
        intermediate_value = dy.tanh(intermediate_value)

        scores = dy.transpose(self.final_weights) * intermediate_value + self.final_biases
        return scores

    def evaluate(self, feature, label):
        prediction = np.argmax(self.get_score(feature).value())
        return prediction == label

    def epoch_train(self, train_input, train_label, batch_size):
        start_time = time.time()
        dy.renew_cg()
        indices = [i for i in range(len(train_input))]
        random.shuffle(indices)
        current_losses = [ ]
        for index in indices:
            loss = dy.pickneglogsoftmax(self.get_score(train_input[index]), train_label[index])
            current_losses.append(loss)

            if len(current_losses) >= batch_size:
                mean_loss = dy.esum(current_losses) / float(len(current_losses))
                mean_loss.forward()
                mean_loss.backward()
                self.optimizer.update()
                current_losses = [ ]
                dy.renew_cg()

        if current_losses:
            mean_loss = dy.esum(current_losses) / float(len(current_losses))
            mean_loss.forward()
            mean_loss.backward()
            self.optimizer.update()
        print("total time: " + str(time.time() - start_time))

    def train(self, train_input, train_label, batch_size, epoch_num):
        """ Trains the maximum entropy model.

        Inputs:
            training_data: Suggested type is (list of pair), where each item is
                a training example represented as an (input, label) pair.
        """
        # Optimize the model using the training data.
        # TODO: Implement the training of this model.
        max_accuracy = 0.
        best_epoch = 0
        start_time = time.time()
        for i in range(epoch_num):
            self.epoch_train(train_input, train_label, batch_size)
            accuracy = sum([float(self.evaluate(train_input[i], train_label[i])) for i in range(len(train_input))]) / float(len(train_input))
            print("epoch " + str(i) + " accuracy: " + str(accuracy))
            if accuracy > max_accuracy:
                print("improved!")
                self.pc.save("model-epoch" + str(i) + ".dy")
                best_epoch = i
                max_accuracy = accuracy

        total_time = time.time() - start_time
        print("total training time: " + str(total_time) + "; " + str(float(total_time) / epoch_num) + " per epoch")
        print("loading from model at epoch " + str(best_epoch))
        self.pc.populate("model-epoch" + str(best_epoch) + ".dy")

    def predict(self, dev_input):
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example, represented as a
                feature vector.

        Returns:
            The predicted class.

        """
        # TODO: Implement prediction for an input.
        predictions = []
        for feature in dev_input:
            prediction = np.argmax(self.get_score(feature).value())
            predictions.append(prediction)
        return predictions

if __name__ == "__main__":
    train_data, dev_data, test_data, data_type = load_data(sys.argv)

    # Train the model using the training data.
    model = MultilayerPerceptronModel()
    batch_size = 10
    epoch_num = 25
    model.train(train_data[0], train_data[1], batch_size, epoch_num)
    pred = model.predict(dev_data[0])

    # Predict on the development set.
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", "mlp_" + data_type + "_dev_predictions.csv"))

    # Predict on the test set.
    # Note: We don't provide labels for test, so the returned value from this
    # call shouldn't make sense.
    evaluate(model,
             test_data,
             os.path.join("results", "mlp_" + data_type + "_test_predictions.csv"))
