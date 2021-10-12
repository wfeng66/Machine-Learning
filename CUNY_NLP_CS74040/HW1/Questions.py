import numpy as np
import pandas as pd
import preprocessing as pp
import pickle
from tqdm import tqdm

path = "G://CUNY/NLP/Assignments/HW1/"
# load data set
train_l, test_l = pp.load_data(path, "train.txt", "test.txt")

# tokenization
# train_tkn_l = pp.token(train_l, test_l)
# tr_tkn_no_unk_f = open(path+'tr_tkn_no_unk.txt', 'wb')
# pickle.dump(train_tkn_l, tr_tkn_no_unk_f)
# tr_tkn_no_unk_f.close()

# replace words occurring once with '<unk>'
# train_tkn_unk_l = pp.mark_training_unk(train_tkn_l)
# tr_tkn_unk_f = open(path+'tr_tkn_unk.txt', 'wb')
# pickle.dump(train_tkn_unk_l, tr_tkn_unk_f)
# tr_tkn_unk_f.close()


tr_tkn_unk_f = open(path+'tr_tkn_unk.txt', 'rb')
train_tkn_unk_l = pickle.load(tr_tkn_unk_f)
tr_tkn_unk_f.close()

# question 1
voc_unk = pp.creat_vocabulary(train_tkn_unk_l)
print('The are {} word types in the training corpus.'.format(len(voc_unk)-1))

# question 2
tr_tkn_unk_l = train_tkn_unk_l.copy()
tr_tkn_unk_l.remove('<s>')
print('The are {} word tokens in the training corpus.'.format(len(tr_tkn_unk_l)))

# question 3
train_tkn_l = pp.token(train_l)
train_tkn_l.remove('<s>')
tr_voc_no_unk = pp.creat_vocabulary(train_tkn_l)
test_tkn_l = pp.token(test_l)
test_tkn_l.remove('<s>')
no_seen_tkn = 0
for wd in test_tkn_l:
    if wd not in tr_voc_no_unk:
        no_seen_tkn += 1
print("{:.4%} of word tokens in the test corpus did not occur in training.".format(no_seen_tkn/len(test_tkn_l)))
no_seen_voc = 0
test_voc_no_unk = pp.creat_vocabulary(test_tkn_l)
for wd in test_voc_no_unk:
    if wd not in tr_voc_no_unk:
        no_seen_voc += 1
print("{:.4%} of word types in the test corpus did not occur in training.".format(no_seen_voc/len(test_voc_no_unk)))

# question 4
# test_tkn_l.remove('<s>')
# tr_bigramDict_unk = pp.create_bigramDict(train_tkn_unk_l)
# test_tkn_unk_l = [wd if wd in set(tr_tkn_unk_l) else '<unk>' for wd in tqdm(test_tkn_l)]
# test_bigramDict_unk = pp.create_bigramDict(test_tkn_unk_l)
# test_tkn_unk_f = open(path+'test_tkn_unk.txt', 'wb')
# pickle.dump(test_tkn_unk_l, test_tkn_unk_f)
# test_tkn_unk_f.close()
# tr_bigramDict_unk_f = open(path+'tr_bigramDict_unk.txt', 'wb')
# pickle.dump(tr_bigramDict_unk, tr_bigramDict_unk_f)
# tr_bigramDict_unk_f.close()
# test_bigramDict_unk_f = open(path+'test_bigramDict_unk.txt', 'wb')
# pickle.dump(test_bigramDict_unk, test_bigramDict_unk_f)
# test_bigramDict_unk_f.close()
test_bigramDict_unk_f = open(path+'test_bigramDict_unk.txt', 'rb')
test_bigramDict_unk = pickle.load(test_bigramDict_unk_f)
test_bigramDict_unk_f.close()
tr_bigramDict_unk_f = open(path+'tr_bigramDict_unk.txt', 'rb')
tr_bigramDict_unk = pickle.load(tr_bigramDict_unk_f)
tr_bigramDict_unk_f.close()
no_seen_bigram = 0
for bg in test_bigramDict_unk.keys():
    if bg not in tr_bigramDict_unk.keys():
        no_seen_bigram += 1
print('{:.4%} of bigrams in the test corpus did not occur in training.'.format(no_seen_bigram/len(test_bigramDict_unk)))