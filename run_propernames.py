import sys
import numpy as np

from propername import *
from perceptron import PerceptronModel
from maximum_entropy import MaximumEntropyModel
from multilayer_perceptron import MultilayerPerceptronModel
from util import save_results

def main(args):
    train_data_filename = './data/propernames/train/train_data.csv'
    train_labels_filename = './data/propernames/train/train_labels.csv'
    dev_data_filename = './data/propernames/dev/dev_data.csv'
    dev_labels_filename = './data/propernames/dev/dev_labels.csv'
    test_data_filename = './data/propernames/test/test_data.csv'
    test_labels_filename = ''

    train, dev, test, label_map = propername_data_loader(train_data_filename,
                       train_labels_filename,
                       dev_data_filename,
                       dev_labels_filename,
                       test_data_filename,
                       lower=1,
                       upper=4)

    if 'perceptron' in args:
        print('Start training perceptron with proper name data set.')
        per = PerceptronModel(epoch=100, lr=0.01)
        per.train(train)

        dev_data, dev_label = dev
        y_dev_pred = per.predict(dev_data)
        y_dev = np.where(dev_label == 1)[1]
        count = 0
        for i in range(len(y_dev)):
            if y_dev[i] == y_dev_pred[i]:
                count += 1
        print('Dev accuracy: %f' %(count/len(y_dev)))

        test_data, _ = test
        y_test = per.predict(test_data)
        res = []
        for i in range(len(y_test)):
            res.append(label_map[y_test[i]])
        save_results(res, './tmp/perceptron_propername_test_predictions.csv')
        print('----------------------------------------------------------')

    if 'maxent' in args:
        print('Start training MaxEnt with proper name data set.')
        maxent = MaximumEntropyModel(alpha=1)
        maxent.train(train)

        dev_data, dev_label = dev
        y_dev_pred = maxent.predict(dev_data)
        y_dev = np.where(dev_label == 1)[1]
        count = 0
        for i in range(len(y_dev)):
            if y_dev[i] == y_dev_pred[i]:
                count += 1
        print('Dev accuracy: %f' %(count/len(y_dev)))

        test_data, _ = test
        y_test = maxent.predict(test_data)
        res = []
        for i in range(len(y_test)):
            res.append(label_map[y_test[i]])
        save_results(res, './tmp/maxent_propername_test_predictions.csv')
        print('----------------------------------------------------------')

    if 'mlp' in args:
        print('Start training multi layer perceptron with proper name data set.')
        train_data, train_label = train
        train_label = np.where(train_label == 1)[1]
        mlp = MultilayerPerceptronModel(train_data.shape[1], 5)
        mlp.train(train_data, train_label, 16, 10)

        dev_data, dev_label = dev
        y_dev_pred = mlp.predict(dev_data)
        y_dev = np.where(dev_label == 1)[1]
        count = 0
        for i in range(len(y_dev)):
            if y_dev[i] == y_dev_pred[i]:
                count += 1
        print('Dev accuracy: %f' %(count/len(y_dev)))

        test_data, _ = test
        y_test = mlp.predict(test_data)
        res = []
        for i in range(len(y_test)):
            res.append(label_map[y_test[i]])
        save_results(res, './tmp/mlp_propername_test_predictions.csv')
        print('----------------------------------------------------------')

    print('Finish all training tasks.')

if __name__ == "__main__":
    main(sys.argv)
