import numpy as np
import pandas as pd

def load_data(path, tr, ts):
    # load data set
    f_train = open(path+tr, encoding='utf-8')
    f_test = open(path+ts, encoding='utf-8')
    train_l = f_train.read().strip().split("\n")
    test_l = f_test.read().strip().split("\n")
    return train_l, test_l

def token(train_l, padding = 1):
    # pre-processing
    if padding == 2:
        train_l = ['<s> ' + s.lower() + ' </s>' for s in train_l]
    elif padding == 1:
        train_l = [s.lower() + ' </s>' for s in train_l]
    else:
        pass
    train_tkn_l = [s.split(' ') for s in train_l]
    # train_tkn_l = [token for sent in train_tkn_l for token in sent]
    return train_tkn_l

def creat_vocabulary(train_tkn_l):
    train_tkn_l = [token for sent in train_tkn_l for token in sent]
    return set(train_tkn_l)

def mark_training_unk(train_tkn_l, test_tkn_l, test=False):
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
    bigramDict = {}
    for sent in data:
        for i in range(len(sent) - 1):
            if (sent[i], sent[i+1]) in bigramDict:
                bigramDict[(sent[i], sent[i+1])] +=1
            else:
                bigramDict[(sent[i], sent[i + 1])] = 1
    return bigramDict