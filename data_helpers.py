import numpy as np
import re
import itertools
from collections import Counter

def fit_transform(x_test, max_document_length):
    for x in x_test:
        while len(x) < max_document_length :
            add = [0 for i in range(6)]
            x.append(add)
    return x_test

def load_data_and_labels(data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    y, x_test =[], []
    examples = open(data_file, "r", encoding='utf-8').readlines()
    #examples = [s.strip() for s in examples]
    # Split by words
    for k in examples :
        y_data = k.split(',')[-1].strip()
        y.append(y_data)
        k_raw = [list(map(int, i.split())) for i in k.split(',')[:-1]]
        x_test.append(k_raw)
    category = list(set(y))
    category.sort()
    return [x_test, y, category]


def transform_labels(label, category):
    #Generate labels
    labels = []
    for i in range(len(label)):
        element = [0 for x in range(len(category))]
        element[category.index(label[i])] = 1
        labels.append(element)
    return labels

def transform_labels_binary(label, category, kinds):
    labels = []
    for i in range(len(label)):
        if label[i] == category[kinds]:
            labels.append([1,0])
        else:
            labels.append([0,1])
    return labels

def depart_data(data_file):
    examples = np.array(open(data_file, "r", encoding='utf-8').readlines())
    shuffle_indices = np.random.permutation(np.arange(len(examples)))
    examples = examples[shuffle_indices]
    eval_sample_index = -1 * int(0.1 * float(len(examples)))
    data = examples[eval_sample_index:]
    open('./cmm_path/dev.txt', "w", encoding='utf-8').writelines(data)
    data = examples[:eval_sample_index]
    open('./cmm_path/cmm_paths.txt', "w", encoding='utf-8').writelines(data)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    depart_data('./cmm_path/cmm_paths.txt')






