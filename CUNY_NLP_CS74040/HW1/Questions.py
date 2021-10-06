import numpy as np
import pandas as pd

# load data set
f_train = open("train.txt", encoding='utf-8')
f_test = open("test.txt", encoding='utf-8')

train_l = f_train.read().strip().split("\n")
test_l = f_test.read().strip().split("\n")

# pre-processing
train_l = ['<s> '+ s.lower() + ' </s>' for s in train_l]
test_l = ['<s> '+ s.lower() + ' </s>' for s in test_l]

train_l = [s.split(' ') for s in train_l]
train_l = [token for sent in train_l for token in sent]

V = set(train_l)

print(V)

