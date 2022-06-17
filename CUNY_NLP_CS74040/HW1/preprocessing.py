import numpy as np
import pandas as pd

def load_data(path, tr, ts):
    # load data set
    # parameters:   path - string, path store the files
    #               tr   - string, training corpus file name
    #               ts   - string, test corpus file name
    # return 2 list, training and test
    f_train = open(path+tr, encoding='utf-8')
    f_test = open(path+ts, encoding='utf-8')
    train_l = f_train.read().strip().split("\n")
    test_l = f_test.read().strip().split("\n")
    return train_l, test_l

def token(train_l, padding = 1):
    # tokenization
    # parameters:   train_l - list, each element is a string of sentence
    #               padding - the padding approach
    #                           1: only pad '</s>' at the end of the sentence
    #                           2: pad '<s>' at the beginning and '</s>' at the end of the sentence
    # return a 2d list, each row corresponding to a sentence
    # lowercase and padding
    if padding == 2:
        train_l = ['<s> ' + s.lower() + ' </s>' for s in train_l]
    elif padding == 1:
        train_l = [s.lower() + ' </s>' for s in train_l]
    else:
        pass
    train_tkn_l = [s.split(' ') for s in train_l]
    return train_tkn_l

def creat_vocabulary(train_tkn_l):
    # train_tkn_l - a 2d list including tokenized corpus
    # return a set type of vocabulary, includes all unique data type included in input corpus
    train_tkn_l = [token for sent in train_tkn_l for token in sent]
    return set(train_tkn_l)

def mark_training_unk(train_tkn_l, test_tkn_l, test=False):
    # this function replace the once word with '<unk>'
    # parameters:   train_tkn_l - tokenized 2d training list
    #               test_tkn_l  - tokenized 2d test list
    #               test        - if replace the test list
    #                               True:  replace
    #                               False: don't replace
    # return two list with '<unk>'. If test=False, the test_tkn_l is same as the input
    # find the words occurring once in the training data
    from collections import Counter
    tr_tkn_l_1d = [token for sent in train_tkn_l for token in sent]
    train_tkn_cnt = Counter(tr_tkn_l_1d)
    once_tkn = [w for w, count in train_tkn_cnt.items() if count == 1]
    # replace the words occurring once in the training data with <unk>
    from tqdm import tqdm
    for i in tqdm(range(len(train_tkn_l))):
        for j in range(len(train_tkn_l[i])):
            if train_tkn_l[i][j] in once_tkn:
                train_tkn_l[i][j] = '<unk>'
    if test:
        for i in tqdm(range(len(test_tkn_l))):
            for j in range(len(test_tkn_l[i])):
                if test_tkn_l[i][j] in once_tkn:
                    test_tkn_l[i][j] = '<unk>'
    return train_tkn_l, test_tkn_l


def create_bigramDict(data):
    # create bigram dictionary
    bigramDict = {}         # bigram dictionary
    for sent in data:       # iterate sentences
        for i in range(len(sent) - 1):      # iterate tokens
            if (sent[i], sent[i+1]) in bigramDict:      # the token pair exist in bigramDict
                bigramDict[(sent[i], sent[i+1])] +=1
            else:                                       # new token pair
                bigramDict[(sent[i], sent[i + 1])] = 1
    return bigramDict