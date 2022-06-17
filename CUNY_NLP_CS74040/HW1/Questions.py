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

# Following comment lines are used for tokenization and replace once words, and save the results as files
# If you have the dump files, you don't need them
# tokenization and save the token files
# train_tkn_l = pp.token(train_l)
# tr_tkn_no_unk_f = open(path+'tr_tkn_no_unk.txt', 'wb')
# pickle.dump(train_tkn_l, tr_tkn_no_unk_f)
# tr_tkn_no_unk_f.close()
# test_tkn_l = pp.token(test_l)
# test_tkn_no_unk_f = open(path+'test_tkn_no_unk.txt', 'wb')
# pickle.dump(test_tkn_l, test_tkn_no_unk_f)
# test_tkn_no_unk_f.close()

# replace words occurring once with '<unk>' and save the results
# train_tkn_unk_l, test_tkn_unk_l = pp.mark_training_unk(train_tkn_l, test_tkn_l, test=True)
# tr_tkn_unk_f = open(path+'tr_tkn_unk.txt', 'wb')
# pickle.dump(train_tkn_unk_l, tr_tkn_unk_f)
# tr_tkn_unk_f.close()
# test_tkn_unk_f = open(path+'test_tkn_unk.txt', 'wb')
# pickle.dump(test_tkn_unk_l, test_tkn_unk_f)
# test_tkn_unk_f.close()

# load the training token with <unk> mark saved before
tr_tkn_unk_f = open(path+'tr_tkn_unk.txt', 'rb')
train_tkn_unk_l = pickle.load(tr_tkn_unk_f)
tr_tkn_unk_f.close()


# question 1
voc_unk = pp.creat_vocabulary(train_tkn_unk_l)          # create vocabulary by training tokens with '<unk>' mark
print('\nQuestion 1:\n'+'-'*50)
print('There are {} word types in the training corpus.'.format(len(voc_unk)-1))

# question 2
print('\nQuestion 2:\n'+'-'*50)
train_tkn_l = pp.token(train_l)                                     # tokenization
train_tkn_l = [token for sent in train_tkn_l for token in sent]     # convert 2d list to 1d list
print('There are {} word tokens in the training corpus.'.format(len(train_tkn_l)))

# question 3
print('\nQuestion 3:\n'+'-'*50)
train_l, test_l = pp.load_data(path, "train.txt", "test.txt")
train_tkn_l = pp.token(train_l)                                     # tokenization
train_tkn_l = [token for sent in train_tkn_l for token in sent]     # convert 2d list to 1d list
tr_voc_no_unk = set(train_tkn_l)                                    # create vocabulary
test_tkn_l = pp.token(test_l)                                       # tokenization
test_tkn_l = [token for sent in test_tkn_l for token in sent]       # convert 2d list to 1d list
no_seen_tkn = 0
# count the number of words haven't seen in training corpus
for wd in test_tkn_l:
    if wd not in tr_voc_no_unk:
        no_seen_tkn += 1
print("{:.4%} of word tokens in the test corpus did not occur in training.".format(no_seen_tkn/len(test_tkn_l)))
no_seen_voc = 0
# count the number of word types haven't seen in training corpus
test_voc_no_unk = set(test_tkn_l)
for wd in test_voc_no_unk:
    if wd not in tr_voc_no_unk:
        no_seen_voc += 1
print("{:.4%} of word types in the test corpus did not occur in training.".format(no_seen_voc/len(test_voc_no_unk)))

# question 4
print('\nQuestion 4:\n'+'-'*50)
# load data
train_l, test_l = pp.load_data(path, "train.txt", "test.txt")
# load training token with <unk>
tr_tkn_unk_f = open(path+'tr_tkn_unk.txt', 'rb')
train_tkn_unk_l = pickle.load(tr_tkn_unk_f)
tr_tkn_unk_f.close()
test_tkn_unk_f = open(path+'test_tkn_unk.txt', 'rb')
test_tkn_unk_l = pickle.load(test_tkn_unk_f)
test_tkn_unk_f.close()
# creat bigram dictionaries for training and test data sets
tr_bigramDict_unk = pp.create_bigramDict(train_tkn_unk_l)
test_bigramDict_unk = pp.create_bigramDict(test_tkn_unk_l)
# save bigramDicts
# tr_bigramDict_unk_f = open(path+'tr_bigramDict_unk.txt', 'wb')
# pickle.dump(tr_bigramDict_unk, tr_bigramDict_unk_f)
# tr_bigramDict_unk_f.close()
# test_bigramDict_unk_f = open(path+'test_bigramDict_unk.txt', 'wb')
# pickle.dump(test_bigramDict_unk, test_bigramDict_unk_f)
# test_bigramDict_unk_f.close()

# load bigram dictionary saved before
test_bigramDict_unk_f = open(path+'test_bigramDict_unk.txt', 'rb')
test_bigramDict_unk = pickle.load(test_bigramDict_unk_f)
test_bigramDict_unk_f.close()
tr_bigramDict_unk_f = open(path+'tr_bigramDict_unk.txt', 'rb')
tr_bigramDict_unk = pickle.load(tr_bigramDict_unk_f)
tr_bigramDict_unk_f.close()
no_seen_bigram = 0
# count the number of bigrams did not occur in training corpus
for bg in test_bigramDict_unk.keys():
    if bg not in tr_bigramDict_unk.keys():
        no_seen_bigram += 1
print('{:.4%} of bigrams in the test corpus did not occur in training.'.format(no_seen_bigram/len(test_bigramDict_unk)))

# question 5&6
print('\nQuestion 5 and 6:\n'+'-'*50)
sent = 'I look forward to hearing your reply .'.split(' ')
print(sent)
# load the tokens saved before
tr_tkn_nopad_unk_f = open(path+'tr_tkn_nopad_unk.txt', 'rb')
tr_tkn_nopad_unk_l = pickle.load(tr_tkn_nopad_unk_f)
tr_tkn_nopad_unk_f.close()
# mark <unk> for sent
from collections import Counter
tr_tkn_nopad_l = [s.split(' ') for s in train_l]
tr_tkn_l_1d = [token for sent in tr_tkn_nopad_l for token in sent]
train_tkn_cnt = Counter(tr_tkn_l_1d)
once_tkn = [w for w, count in train_tkn_cnt.items() if count == 1]
for j in range(len(sent)):
    if sent[j] in once_tkn:
        sent[j] = '<unk>'

# unigram model
print('Unigram:')
uniModel = model.unigram(tr_tkn_nopad_unk_l)
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
biModel = model.bigram(tr_tkn_nopad_unk_l)
biModel.fit()
sum_p = 0
unk = False
for i in range(len(sent)-1):
    if i == 0:
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
smModel = model.smoothing1(tr_tkn_nopad_unk_l)
smModel.fit()
sum_p = 0
tr_tkn_nopad_l_1d = [token for sent in tr_tkn_nopad_l for token in sent]
n_V = len(set(tr_tkn_nopad_l_1d))
for i in range(len(sent)-1):
    if i == 0:
        if sent[i] == '<unk>':
            log_p = m.log2(1/+n_V)
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

# Bigram with discounting and Katz backoff
print('\nBigram with discounting and Katz backoff:')
# katzModel = model.katz(tr_tkn_nopad_l)
# katzModel.fit()
# save the model
# katzModel_f = open(path+'katzModel.txt', 'wb')
# pickle.dump(katzModel, katzModel_f)
# katzModel_f.close()
# load the katz model saved before
katzModel_f = open(path+'katzModel_unk.txt', 'rb')
katzModel = pickle.load(katzModel_f)
katzModel_f.close()
sum_p = 0
tr_tkn_nopad_l_1d = [token for sent in tr_tkn_nopad_l for token in sent]
n_V = len(set(tr_tkn_nopad_l_1d))
for i in range(len(sent)-1):
    if i == 0:
        if sent[i] == '<unk>':
            log_p = m.log2(1/+n_V)
        else:
            log_p = m.log2(katzModel.p1[sent[i]])
        print('p({0}) = {1:.6f},'.format(sent[i], log_p))
    else:
        if (sent[i], sent[i+1]) in katzModel.p2:
            log_p = m.log2(katzModel.p2[(sent[i], sent[i+1])])
        else:
            log_p = m.log2(katzModel.a.get(sent[i])*katzModel.p1.get(sent[i+1]))
        print('p({0}|{1}) = {2:.6f},'.format(sent[i + 1], sent[i], log_p))
    sum_p += log_p
print('The log probability of "I look forward to hearing your reply ." ')
print('under Bigram with with discounting and Katz backoff is {:.6f}'.format(sum_p))
l = sum_p/len(sent)
print('The perplexity is ', pow(2, -l))

# Question 7
print('\nQuestion 7:\n'+'-'*50)

# train_l, test_l = pp.load_data(path, "train.txt", "test.txt")
# test_tkn_nopad_l = [sent.split(' ') for sent in test_l]
# tr_tkn_nopad_l = [s.split(' ') for s in train_l]        # tokenization
# mark the once word in training corpus as <unk>
# tr_tkn_nopad_unk_l, test_tkn_nopad_unk_l = pp.mark_training_unk(tr_tkn_nopad_l, test_tkn_nopad_l, test=True)
# tr_voc_nopad_unk = pp.creat_vocabulary(tr_tkn_nopad_unk_l)
# replace '<unk>' in test corpus
# for sent in test_tkn_nopad_unk_l:
#     for i in range(len(sent)):
#         if sent[i] not in tr_voc_nopad_unk:
#             sent[i] = '<unk>'
# save the nopade with <unk> mark tokens
# tr_tkn_nopad_unk_f = open(path+'tr_tkn_nopad_unk.txt', 'wb')
# pickle.dump(tr_tkn_nopad_unk_l, tr_tkn_nopad_unk_f)
# tr_tkn_nopad_unk_f.close()
# test_tkn_nopad_unk_f = open(path+'test_tkn_nopad_unk.txt', 'wb')
# pickle.dump(test_tkn_nopad_unk_l, test_tkn_nopad_unk_f)
# test_tkn_nopad_unk_f.close()

# load the tokens saved before
tr_tkn_nopad_unk_f = open(path+'tr_tkn_nopad_unk.txt', 'rb')
tr_tkn_nopad_unk_l = pickle.load(tr_tkn_nopad_unk_f)
tr_tkn_nopad_unk_f.close()
test_tkn_nopad_unk_f = open(path+'test_tkn_nopad_unk.txt', 'rb')
test_tkn_nopad_unk_l = pickle.load(test_tkn_nopad_unk_f)
test_tkn_nopad_unk_f.close()


# unigram model
print('Unigram:')
uniModel = model.unigram(tr_tkn_nopad_unk_l)
uniModel.fit()
sum_p = 0
for sent in test_tkn_nopad_unk_l:
    for wd in sent:
        log_p = m.log2(uniModel.p1[wd])
        sum_p += log_p
print('The log probability of entire test corpus under Unigram is {:.6f}'.format(sum_p))
l = sum_p/len(pp.creat_vocabulary(test_tkn_nopad_unk_l))
print('The perplexity is ', pow(2, -l))

# Bigram Model
print('\nBigram:')
biModel = model.bigram(tr_tkn_nopad_unk_l)
biModel.fit()
sum_p = 0
unk = False                     # if meet the bigram never seen in training corpus, unk = True
for sent in test_tkn_nopad_unk_l:
    if unk:
        break
    for i in range(len(sent)-1):
        if i == 0:              # The first word in a sentence, use p(w_1) instead of p(w_i|w_(i-1))
            log_p = m.log2(biModel.p1[sent[i]])
        else:                   # subsequent words in a sentence
            if (sent[i], sent[i+1]) in biModel.p2:
                log_p = m.log2(biModel.p2[(sent[i], sent[i+1])])
            else:               # the bigram haven't seen in the training corpus
                unk = True
        if not unk:
            sum_p += log_p
if not unk:
    print('The log probability of entire test corpus under Bigram is {:.6f}'.format(sum_p))
    l = sum_p/len(pp.creat_vocabulary(test_tkn_nopad_unk_l))
    print('The perplexity is ', pow(2, -l))
else:
    print("Because some bigrams don't exist in training corpus, no log probability, total probability is zero.")

# Bigram with Add-1 Smoothing
print('\nBigram with add-1 smoothing:')
smModel = model.smoothing1(tr_tkn_nopad_unk_l)
smModel.fit()
sum_p = 0
tr_voc_nopad_unk = pp.creat_vocabulary(tr_tkn_nopad_unk_l)
n_V = len(tr_voc_nopad_unk)       # the number of word types
for sent in test_tkn_nopad_unk_l:
    for i in range(len(sent)-1):
        if i == 0:
            if sent[i] == '<unk>':
                log_p = m.log2(1/+n_V)
            else:
                log_p = m.log2(smModel.p1[sent[i]])
        else:
            if (sent[i], sent[i+1]) in biModel.p2:
                log_p = m.log2(smModel.p2[(sent[i], sent[i+1])])
            else:
                log_p = m.log2(1/(smModel.unigramDict[sent[i]]+n_V))
        sum_p += log_p
print('The log probability of entire test corpus ')
print('under Bigram with add-a smoothing is {:.6f}'.format(sum_p))
l = sum_p/len(pp.creat_vocabulary(test_tkn_nopad_unk_l))
print('The perplexity is ', pow(2, -l))

# Bigram with discounting and Katz backoff
print('\nBigram with discounting and Katz backoff:')
# katzModel = model.katz(tr_tkn_nopad_unk_l)
# katzModel.fit()
# save the model
# katzModel_f = open(path+'katzModel_unk.txt', 'wb')
# pickle.dump(katzModel, katzModel_f)
# katzModel_f.close()
# load the katz model saved before
katzModel_f = open(path+'katzModel_unk.txt', 'rb')
katzModel = pickle.load(katzModel_f)
katzModel_f.close()
sum_p = 0
n_V = len(set(tr_voc_nopad_unk))
for sent in test_tkn_nopad_unk_l:
    for i in range(len(sent)-1):
        if i == 0:
            if sent[i] == '<unk>':
                log_p = m.log2(1/+n_V)
            else:
                log_p = m.log2(katzModel.p1[sent[i]])
        else:
            if (sent[i], sent[i+1]) in katzModel.p2:
                log_p = m.log2(katzModel.p2[(sent[i], sent[i+1])])
            else:
                log_p = m.log2(katzModel.a.get(sent[i])*katzModel.p1.get(sent[i+1]))
        sum_p += log_p
print('The log probability of entire test corpus ')
print('under Bigram with with discounting and Katz backoff is {:.6f}'.format(sum_p))
l = sum_p/len(pp.creat_vocabulary(test_tkn_nopad_unk_l))
print('The perplexity is ', pow(2, -l))

