import csv
import sys
import numpy as np
import nltk
from nltk.tag import pos_tag
from nltk.util import ngrams
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from perceptron import PerceptronModel
from maximum_entropy import MaximumEntropyModel
from multilayer_perceptron import MultilayerPerceptronModel

stemmer = PorterStemmer()
def tokenize(text):
    return [stemmer.stem(x) for x in word_tokenize(text) if x.isalpha()]

def one_hot(labels, n_classes):
    """ Transform labels into one-hot represenattion.

    Inputs:
        labels: input labels"

    Returns:
        one_hot_labels: one-hot representation.
    """
    # n_classes = len(set(labels))
    label = np.array(labels).reshape(-1)
    return np.eye(n_classes)[label]

def save_results(predictions, results_path):
    """ Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    # TODO: Implement saving of the results.
    with open(results_path, "w", newline = '') as csvfile:
        fieldnames = ['id', 'newsgroup']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(len(predictions)):
            writer.writerow({'id': i, 'newsgroup': predictions[i]})

def main(args):
    train_data_filename = 'train_data.csv'
    train_labels_filename = 'train_labels.csv'
    dev_data_filename = 'dev_data.csv'
    dev_labels_filename = 'dev_labels.csv'
    test_data_filename = 'test_data.csv'
    test_labels_filename = ''

    csv.field_size_limit(1000000000)
    filename_list = [train_data_filename, train_labels_filename, dev_data_filename, dev_labels_filename, test_data_filename]
    data_list={"train": {}, "dev": {}, "test": {}}
    for name in filename_list:
        data_class = name.split("_")[0]
        data_type = "input" if name.split("_")[1].split(".")[0] == "data" else "label"
        fieldname = "text" if data_type == "input" else "newsgroup"
        path = "./data/newsgroups/{0}/{1}".format(data_class, name)
        data_list[data_class][data_type] = []
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
    #             data = data_clean(row[fieldname]) if data_type == "input" else row[fieldname]
                data = row[fieldname] if data_type == "input" else row[fieldname]
                data_list[data_class][data_type].append(data)
    data_list["test"]["label"] = [0 for i in range(len(data_list["test"]["input"]))]


    train_label = np.array(data_list["train"]["label"])
    dev_label = np.array(data_list["dev"]["label"])
    test_label = np.array(data_list["test"]["label"])

    labels = set(train_label)
    labels = list(labels)
    label2val = {}
    val2label = {}
    i = 0
    for label in labels:
        label2val[label] = i
        val2label[i] = label
        i += 1
        
    train_label = list(map(lambda x : label2val[x], train_label))
    dev_label = list(map(lambda x : label2val[x], dev_label))

    train_label = one_hot(train_label, 20)
    dev_label = one_hot(dev_label, 20)

    stopwords = set(nltk.corpus.stopwords.words('english'))

    train = data_list["train"]["input"]
    test = data_list["test"]["input"]
    dev = data_list["dev"]["input"]

    vectorizer = CountVectorizer(ngram_range=(1,2), tokenizer=tokenize, binary=True,max_features=30000, stop_words=stopwords)
    train_ = vectorizer.fit_transform(train).toarray()
    dev_ = vectorizer.transform(dev).toarray()
    test_ = vectorizer.transform(test).toarray()

    if 'perceptron' in args:
        print('Start training perceptron with news groups data set.')
        per = PerceptronModel(epoch=75, lr=0.001)
        per.train([train_, train_label])

        y_dev_pred = per.predict(dev_)
        y_dev = np.where(dev_label == 1)[1]
        count = 0
        for i in range(len(y_dev)):
            if y_dev[i] == y_dev_pred[i]:
                count += 1
        print('Dev accuracy: %f' %(count/len(y_dev)))

        y_test = per.predict(test_)
        res = []
        for i in range(len(y_test)):
            res.append(val2label[y_test[i]])
        save_results(res, './tmp/perceptron_newsgroup_test_predictions.csv')
        print('----------------------------------------------------------')

    if 'maxent' in args:
        print('Start training MaxEnt with news groups data set.')
        maxent = MaximumEntropyModel(alpha=1)
        maxent.train([train_, train_label])

        y_dev_pred = maxent.predict(dev_)
        y_dev = np.where(dev_label == 1)[1]
        count = 0
        for i in range(len(y_dev)):
            if y_dev[i] == y_dev_pred[i]:
                count += 1
        print('Dev accuracy: %f' %(count/len(y_dev)))

        y_test = maxent.predict(test_)
        res = []
        for i in range(len(y_test)):
            res.append(val2label[y_test[i]])
        save_results(res, './tmp/maxent_newsgroup_test_predictions.csv')
        print('----------------------------------------------------------')

    if 'mlp' in args:
        print('Start training multi layer perceptron with news groups data set.')
        train_label = np.where(train_label == 1)[1]
        mlp = MultilayerPerceptronModel(train_.shape[1], 20)
        mlp.train(train_, train_label, 16, 20)

        y_dev_pred = mlp.predict(dev_)
        y_dev = np.where(dev_label == 1)[1]
        count = 0
        for i in range(len(y_dev)):
            if y_dev[i] == y_dev_pred[i]:
                count += 1
        print('Dev accuracy: %f' %(count/len(y_dev)))

        y_test = mlp.predict(test_)
        res = []
        for i in range(len(y_test)):
            res.append(val2label[y_test[i]])
        save_results(res, './tmp/mlp_newsgroup_test_predictions.csv')
        print('----------------------------------------------------------')

    print('Finish all training tasks.')

if __name__ == "__main__":
    main(sys.argv)
