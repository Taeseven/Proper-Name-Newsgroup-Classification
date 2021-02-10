import csv
import nltk
import numpy as np
from nltk.tag import pos_tag

def data_clean(text):
    stopwords = nltk.corpus.stopwords.words('english')
    starters = ['From:', 'Subject:', 'Summary:', 'Organization:', 'Lines:', 
                'Reply-To:', 'NNTP-Posting-Host:', 'Distribution:', 'Keywords: ', 'Article-I.D.:',
                'X-Newsreader:', 'Nntp-Posting-Host:']
    special_characters = ['.', '@','<', '>', '|', '!', '-', '?', ',', '\\', 
                            '/', '(', ')', ';', '{', '}', ':', '#', '$', '%', 
                            '^', '&', '*', '~', '`', '_', '+', '=', '.', '\'',
                            '\"', '[', ']']
    words = []
    for line in text.split('\n'):
        f = True
        for starter in starters:
            if line.startswith(starter):
                f = False
                break
        if not f:
            continue
        for c in special_characters:
            line = line.replace(c, ' ')
        for word in line.lower().split():
            if (word not in stopwords) and (not word.isdigit()) and len(word) != 1:
                words.append(word)
    return words 

def word_count(text_list):
    word_dict = {}
    for text in text_list:
        for word in text:
            word_dict[word] = word_dict.get(word, 0) + 1
    word_dict_sorted = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
    return word_dict_sorted

def bow_noun(word_dict):
    word_list = [x[0] for x in word_dict]
    tag_list = pos_tag(word_list)
    noun_tags = ['NN', 'NNS']
    noun_list = []
    for i in range(len(tag_list)):
        if tag_list[i][1] in noun_tags:
            noun_list.append(tag_list[i][0])
    bow_noun_features = []
    for i in range(len(tag_list)):
        if tag_list[i][1] in noun_tags:
            if word_dict[i][1] < 10:
                break
            bow_noun_features.append(tag_list[i][0])
    return bow_noun_features

def text_to_bow_noun_vector(text, bow_noun_features):
    vector = [0 for i in range(len(bow_noun_features))]
    for word in text:
        if word in bow_noun_features:
            vector[bow_noun_features.index(word)] = 1
    return np.array(vector)

def newsgroup_featurize(data_list):
    """ Featurizes an input for the newsgroup domain.

    Inputs:
        input_data: The input data.
    """
    # TODO: Implement featurization of input.
    all_text = data_list["train"]["input"] + data_list["test"]["input"] + data_list["dev"]["input"]
    word_dict = word_count(all_text)
    bow_noun_features = bow_noun(word_dict) # 11,925 features
    train_input = np.array([text_to_bow_noun_vector(text, bow_noun_features) for text in data_list["train"]["input"]])
    dev_input = np.array([text_to_bow_noun_vector(text, bow_noun_features) for text in data_list["dev"]["input"]])
    test_input = np.array([text_to_bow_noun_vector(text, bow_noun_features) for text in data_list["test"]["input"]])
    return train_input, dev_input, test_input    

def newsgroup_data_loader(train_data_filename,
                          train_labels_filename,
                          dev_data_filename,
                          dev_labels_filename,
                          test_data_filename):
    """ Loads the data.

    Inputs:
        train_data_filename (str): The filename of the training data.
        train_labels_filename (str): The filename of the training labels.
        dev_data_filename (str): The filename of the development data.
        dev_labels_filename (str): The filename of the development labels.
        test_data_filename (str): The filename of the test data.

    Returns:
        Training, dev, and test data, all represented as a list of (input, label) format.

        Suggested: for test data, put in some dummy value as the label.
    """
    # TODO: Load the data from the text format.
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
                data = data_clean(row[fieldname]) if data_type == "input" else row[fieldname]
                data_list[data_class][data_type].append(data)
    data_list["test"]["label"] = [0 for i in range(len(data_list["test"]["input"]))]

    # TODO: Featurize the input data for all three splits.
    train_input, dev_input, test_input = newsgroup_featurize(data_list)
    train_label = np.array(data_list["train"]["label"])
    dev_label = np.array(data_list["dev"]["label"])
    test_label = np.array(data_list["test"]["label"])

    return [train_input, train_label], [dev_input, dev_label], [test_input, test_label]
