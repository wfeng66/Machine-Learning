import numpy as np
import pandas as pd
import math as m
import preprocessing as pp
import model
import pickle
from tqdm import tqdm

path = "G://CUNY/NLP/Assignments/HW1/"
# load data set
train_l, test_l = pp.load_data(path, "train.txt", "test.txt")

# tokenization
# train_tkn_l = pp.token(train_l)
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
# voc_unk = pp.creat_vocabulary(train_tkn_unk_l)
voc_unk = set(train_tkn_unk_l)
print('\nQuestion 1:\n'+'-'*50)
print('The are {} word types in the training corpus.'.format(len(voc_unk)-1))

# question 2
print('\nQuestion 2:\n'+'-'*50)
tr_tkn_unk_l = train_tkn_unk_l.copy()
tr_tkn_unk_l.remove('<s>')
print('The are {} word tokens in the training corpus.'.format(len(tr_tkn_unk_l)))

# question 3
print('\nQuestion 3:\n'+'-'*50)
train_tkn_l = pp.token(train_l)
train_tkn_l = [token for sent in train_tkn_l for token in sent]
train_tkn_l.remove('<s>')
tr_voc_no_unk = set(train_tkn_l)
test_tkn_l = pp.token(test_l)
test_tkn_l = [token for sent in test_tkn_l for token in sent]
test_tkn_l.remove('<s>')
no_seen_tkn = 0
for wd in test_tkn_l:
    if wd not in tr_voc_no_unk:
        no_seen_tkn += 1
print("{:.4%} of word tokens in the test corpus did not occur in training.".format(no_seen_tkn/len(test_tkn_l)))
no_seen_voc = 0
test_voc_no_unk = set(test_tkn_l)
for wd in test_voc_no_unk:
    if wd not in tr_voc_no_unk:
        no_seen_voc += 1
print("{:.4%} of word types in the test corpus did not occur in training.".format(no_seen_voc/len(test_voc_no_unk)))

# question 4
print('\nQuestion 4:\n'+'-'*50)
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

# question 5
print('\nQuestion 5:\n'+'-'*50)
sent = 'I look forward to hearing your reply .'.split(' ')
tr_tkn_nopad_l = [s.split(' ') for s in train_l]        # tokenization
sent = [tkn if tkn not in tr_tkn_nopad_l else '<unk>' for tkn in sent]      # replace '<unk>'
print(sent)
# unigram model
print('Unigram:')
uniModel = model.unigram(tr_tkn_nopad_l)
uniModel.fit()
sum_p = 0
for wd in sent:
    log_p = m.log2(uniModel.p1[wd])
    sum_p += log_p
    print('p({0}) = {1:.6f},'.format(wd, log_p))
print('The log probability of "I look forward to hearing your reply ." under Unigram is {:.6f}'.format(sum_p))
l = sum_p/len(sent)
print('The perplexity is ', pow(2, -l))

# Bigram Model
print('\nBigram:')
biModel = model.bigram(tr_tkn_nopad_l)
biModel.fit()
sum_p = 0
unk = False
for i in range(len(sent)-1):
    if i == 0:
        if sent[i] == '<unk>':
            print("p({}) doesn't exist in training corpus, no log probability.".format(sent[i]))
            unk = True
        else:
            log_p = m.log2(biModel.p1[sent[i]])
        print('p({0}) = {1:.6f},'.format(sent[i], log_p))
    else:
        if (sent[i], sent[i+1]) in biModel.p2:
            log_p = m.log2(biModel.p2[(sent[i], sent[i+1])])
            print('p({0}|{1}) = {2:.6f},'.format(sent[i + 1], sent[i], log_p))
        else:
            print("p({0}|{1}) doesn't exist in training corpus, no log probability.".format(sent[i+1], sent[i]))
            unk = True
    if not unk:
        sum_p += log_p
if not unk:
    print('The log probability of "I look forward to hearing your reply ." under Bigram is {:.6f}'.format(sum_p))
else:
    print("Because some bigrams don't exist in training corpus, no log probability, total probability is zero.")

# Bigram with Add-1 Smoothing
print('\nBigram with add-1 smoothing:')
smModel = model.smoothing1(tr_tkn_nopad_l)
smModel.fit()
sum_p = 0
tr_tkn_nopad_l_1d = [token for sent in tr_tkn_nopad_l for token in sent]
n_V = len(set(tr_tkn_nopad_l_1d))
for i in range(len(sent)-1):
    if i == 0:
        if sent[i] == '<unk>':
            log_p = m.log2(1/n_V)
        else:
            log_p = m.log2(smModel.p1[sent[i]])
        print('p({0}) = {1:.6f},'.format(sent[i], log_p))
    else:
        if (sent[i], sent[i+1]) in biModel.p2:
            log_p = m.log2(smModel.p2[(sent[i], sent[i+1])])
        else:
            log_p = m.log2(1/(smModel.unigramDict[sent[i]]+n_V))
        print('p({0}|{1}) = {2:.6f},'.format(sent[i + 1], sent[i], log_p))
    sum_p += log_p
print('The log probability of "I look forward to hearing your reply ." ')
print('under Bigram with add-a smoothing is {:.6f}'.format(sum_p))
l = sum_p/len(sent)
print('The perplexity is ', pow(2, -l))

