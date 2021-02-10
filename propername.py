import csv
import numpy as np

def propername_featurize(input_data, lower=2, upper=2):
    """ Featurizes an input for the proper name domain.

    Inputs:
        input_data: The input data. [train_data, dev_data, test_data]
    """
    # TODO: Implement featurization of input.
    train_data, dev_data, test_data = input_data
    train_data = ngram_transform(train_data, lower, upper)
    dev_data = ngram_transform(dev_data, lower, upper)
    test_data = ngram_transform(test_data, lower, upper)

    gram_dict = get_ngram_dict(train_data)

    train_data = ngram_fit_transform(train_data, gram_dict)
    dev_data = ngram_fit_transform(dev_data, gram_dict)
    test_data = ngram_fit_transform(test_data, gram_dict)

    return train_data, dev_data, test_data

def ngram_transform(data, l, h):
    """ Transform data into n-gram character string

    Inputs:
        data: input data
        l: lower bound of n
        h: upper bound of n
    
    Returns:
        data_: list of string, concatenate different ngram string together
    """
    data_ = []
    for item in data:
        tmp = list(item[0].lower())
        data_tmp = []
        for n in range(l, h+1):
            data_tmp += [''.join(tmp[i:i+n]) for i in range(len(tmp)-n+1)]
#         print(data_tmp)
        data_.append(data_tmp)
    return data_

def get_ngram_dict(data):
    """ Get ngram dictionary

    Inputs:
        data: list of data (already in ngram shape)
    Returns:
        gram_dict: dictionary from ngram string to idex

    Hint: Please only pass train data into this function
    """
    gram_dict = {}
    i = 0
    for item in data:
        for gram in item:
            if gram not in gram_dict:
                gram_dict[gram] = i
                i += 1
    return gram_dict

def ngram_fit_transform(data, gram_dict):
    """ Transform ngram spring into numerical futures

    Inputs:
        data: ngram string features
        gram_dict: ngram dictionary
    Returns:
        new_data: numerical ngram feature data
    """
    new_data = []
    for item in data:
        tmp = np.zeros(len(gram_dict))
        for gram in item:
            if gram in gram_dict:
                tmp[gram_dict[gram]] += 1
        new_data.append(tmp)
    new_data = np.array(new_data)
    return new_data

def propername_data_loader(train_data_filename,
                           train_labels_filename,
                           dev_data_filename,
                           dev_labels_filename,
                           test_data_filename,
                           lower=2,
                           upper=2):
    """ Loads the data.

    Inputs:
        train_data_filename (str): The filename of the training data.
        train_labels_filename (str): The filename of the training labels.
        dev_data_filename (str): The filename of the development data.
        dev_labels_filename (str): The filename of the development labels.
        test_data_filename (str): The filename of the test data.
        lower: lower bound of n in ngram.
        upper: upper bound of n in ngram.

    Returns:
        Training, dev, and test data, all represented as (input, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.
    # 1. labels -> one hot vector
    # need to return the map from value to specific label
    # train labels
    train_labels_file = open(train_labels_filename, "r")
    reader = csv.reader(train_labels_file)
    train_labels = []
    label_set = set()

    for item in reader:
        if reader.line_num == 1:
            continue
        label_set.add(item[1])
        train_labels.append(item[1])
    train_labels_file.close()

    label_map = {}
    val2label = {}
    label_set = list(label_set)

    for i in range(len(label_set)):
        label_map[label_set[i]] = i
        val2label[i] = label_set[i]
    train_labels = list(map(lambda x : label_map[x], train_labels))
    train_labels = one_hot(train_labels, len(label_set))

    # dev labels
    dev_labels = []
    dev_labels_file = open(dev_labels_filename, "r")
    reader = csv.reader(dev_labels_file)
    for item in reader:
        if reader.line_num == 1:
            continue
        dev_labels.append(item[1])
    train_labels_file.close()
    
    dev_labels = list(map(lambda x : label_map[x], dev_labels))
    dev_labels = one_hot(dev_labels, len(label_set))

    # 2. read data
    train_data, test_data, dev_data = [], [], []
    train_data = read_csv(train_data_filename)
    test_data = read_csv(test_data_filename)
    dev_data = read_csv(dev_data_filename)

    # generate test labels according to the shape of test data
    test_labels = np.zeros((len(test_data), len(label_set)))

    # TODO: Featurize the input data for all three splits.
    # 3. featurization
    train_data, dev_data, test_data = propername_featurize([train_data, dev_data, test_data], lower, upper)

    return [train_data, train_labels], [dev_data, dev_labels], [test_data, test_labels], val2label

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

def read_csv(file_name):
    """ Read data

    Inputs:
        file_name: csv file name

    Returns:
        result: list of data
    """
    read_file = open(file_name, "r")
    reader = csv.reader(read_file)
    result = []
    for item in reader:
        if reader.line_num == 1:
            continue
        result.append(item[1:])
    read_file.close()
    return result