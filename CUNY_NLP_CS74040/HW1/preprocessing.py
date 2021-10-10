import numpy as np
import pandas as pd

def load_data(path, tr, ts):
    # load data set
    f_train = open(path+tr, encoding='utf-8')
    f_test = open(path+ts, encoding='utf-8')
    train_l = f_train.read().strip().split("\n")
    test_l = f_test.read().strip().split("\n")
    return train_l, test_l

def token(train_l, test_l):
    # pre-processing
    train_l = ['<s> '+ s.lower() + ' </s>' for s in train_l]
    test_l = ['<s> '+ s.lower() + ' </s>' for s in test_l]
    train_tkn_l = [s.split(' ') for s in train_l]
    train_tkn_l = [token for sent in train_tkn_l for token in sent]
    return train_tkn_l

def creat_vocabulary(train_tkn_l):
    return set(train_tkn_l)

def mark_training_unk(train_tkn_l):
    # find the words occurring once in the training data
    from collections import Counter
    train_tkn_cnt = Counter(train_tkn_l)
    once_tkn = [w for w, count in train_tkn_cnt.items() if count == 1]
    # replace the words occurring once in the training data with <unk>
    from tqdm import tqdm
    for wd in tqdm(once_tkn):
        train_l = [sent.replace(wd, "<unk>") for sent in train_l]
    return train_l
