from propername import propername_data_loader
from newsgroup import newsgroup_data_loader
import csv
import numpy as np

def save_results(predictions, results_path):
    """ Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    # TODO: Implement saving of the results.
    with open(results_path, "w", newline = '') as csvfile:
        fieldnames = ['id', 'type']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(len(predictions)):
            writer.writerow({'id': i, 'type': predictions[i]})
    print('Finish writing predictions to desire csv file.')

def compute_accuracy(labels, predictions):
    """ Computes the accuracy given some predictions and labels.

    Inputs:
        labels (list): Labels for the examples.
        predictions (list): The predictions.
    Returns:
        float representing the % of predictions that were true.
    """
    if len(labels) != len(predictions):
        raise ValueError("Length of labels (" + str(len(labels)) + " not the same as " \
                         "length of predictions (" + str(len(predictions)) + ".")
    # TODO: Implement accuracy computation.
    y_true = np.where(labels == 1)[1]
    count = 0
    for i in range(len(labels)):
        if predictions[i] == y_true[i]:
            count += 1

    return count / len(y_true)

def evaluate(model, data, results_path):
    """ Evaluates a dataset given the model.

    Inputs:
        model: A model with a prediction function.
        data: Suggested type is (list of pair), where each item is a training
            examples represented as an (input, label) pair. And when using the
            test data, your label can be some null value.
        results_path (str): A filename where you will save the predictions.
    """
    feature, label = data
    predictions = model.predict(feature)
    
    acc = compute_accuracy(label, predictions)

    # no need for evaluation in dev data
    # # Transform numbers to labels before
    # save_results(res, results_path)

    return acc

def load_data(args):
    """ Loads the data.

    Inputs:
        args (list of str): The command line arguments passed into the script.

    Returns:
        Training, development, and testing data, as well as which kind of data
            was used.
    """
    data_loader = None
    data_type = ""
    if 'propername' in args:
      data_loader = propername_data_loader
      data_type = "propernames"
    elif 'newsgroup' in args:
      data_loader = newsgroup_data_loader
      data_type = "newsgroups"
    assert data_loader, "Choose between newsgroups or propernames data. " \
                        + "Args was: " + str(args)

    # Load the data. 
    train_data, dev_data, test_data, label_map = data_loader("./data/" + data_type + "/train/train_data.csv",
                                                  "./data/" + data_type + "/train/train_labels.csv",
                                                  "./data/" + data_type + "/dev/dev_data.csv",
                                                  "./data/" + data_type + "/dev/dev_labels.csv",
                                                  "./data/" + data_type + "/test/test_data.csv")

    return train_data, dev_data, test_data, data_type, label_map
