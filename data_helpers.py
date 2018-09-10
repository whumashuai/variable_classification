import numpy as np
import random
import re
import itertools
from collections import Counter
import remove_null

def fit_transform(x_test, max_document_length, embedding_dim):
    for x in x_test:
        while len(x) < max_document_length :
            add = [0 for i in range(embedding_dim)]
            x.append(add)
    return x_test

def load_data_and_labels(data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    y, x_text =[], []
    examples = open(data_file, "r", encoding='utf-8').readlines()
    #examples = [s.strip() for s in examples]
    # Split by words
    for k in examples :
        y_data = k.split(',')[-1].strip()
        y.append(y_data)
        k_raw = [list(map(int, i.split())) for i in k.split(',')[:-1]]
        x_text.append(k_raw)
    category = list(set(y))
    category.sort()
    return [x_text, y, category]

def load_data_and_labels_sampling(data_file, kind, Sample_proportion):
    # Load data from files
    y, x_text = [], []
    examples = open(data_file, "r", encoding='utf-8').readlines()
    # examples = [s.strip() for s in examples]
    # Split by words
    for k in examples:
        y_data = k.split(',')[-1].strip()
        y.append(y_data)
    category = list(set(y))
    category.sort()
    y = []
    if kind >= len(category):
        raise ValueError
    positive,negative = [],[]
    for j in examples:
        if category[kind] in j:
            positive.append(j)
        else:
            negative.append(j)
    if len(negative) <= len(positive)*Sample_proportion:
        proportion = float(len(positive)/len(negative))
        positive.extend(negative)
    else:
        proportion = Sample_proportion
        positive.extend(random.sample(negative, len(positive) * Sample_proportion))
    random.shuffle(positive)
    for m in positive:
        y_data = m.split(',')[-1].strip()
        y.append(y_data)
        k_raw = [list(map(int, i.split())) for i in m.split(',')[:-1]]
        x_text.append(k_raw)
    category = list(set(y))
    category.sort()
    return [x_text, y, category, proportion]


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
    random.shuffle(examples)
    eval_sample_index = -1 * int(0.1 * float(len(examples)))
    data = examples[eval_sample_index:]
    file = data_file.split('.')[0]
    dev_file = file + '_dev.csv'
    open(dev_file, "w", encoding='utf-8').writelines(data)
    data = examples[:eval_sample_index]
    train_file = file + '_train.csv'
    open(train_file, "w", encoding='utf-8').writelines(data)
    return train_file, dev_file


def traceHelper(data_file):
    print(data_file)
    output = data_file +'.csv'
    examples = open(data_file, 'r', encoding='utf-8').readlines()
    for line in examples:
        example = list(filter(len, map(str.strip, line.split(','))))
        if len(example) < 2:
            continue
        open(output, 'a').write(",".join(example[1:]) + ',' + example[0] + '\n')
    return output

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
    #default path="F:\variable_classification\data\opencv\opencv_test_core.traces"
    #data_file = 'F:\\variable_classification\\data\\opencv\\opencv_test_core.traces'
    #data_file = traceHelper(data_file)
    #print(data_file)
    #depart_data(data_file)
    [x_test, y, category] = load_data_and_labels('./data/opencv/opencv_test_core.traces.csv')
    print(len(x_test))
    print(len(y))
    max_document_length = max([len(x) for x in x_test])
    print(max_document_length)
    print(category)
    print(x_test[0])

    for label in category:
        print("%s,%d"%(label,y.count(label)))




