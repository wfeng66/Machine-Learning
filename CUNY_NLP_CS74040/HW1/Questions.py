import numpy as np
import pandas as pd
import preprocessing as pp
import pickle

path = "G://CUNY/NLP/Assignments/HW1/"
# load data set
train_l, test_l = pp.load_data(path, "train.txt", "test.txt")

# tokenization
train_tkn_l = pp.token(train_l, test_l)
tr_tkn_no_unk_f = open(path+'tr_tkn_no_unk.txt', 'wb')
pickle.dump(train_tkn_l, tr_tkn_no_unk_f)
tr_tkn_no_unk_f.close()

# replace words occurring once with '<unk>'
train_tkn_unk_l = pp.mark_training_unk(train_tkn_l)
tr_tkn_unk_f = open(path+'tr_tkn_unk.txt', 'wb')
pickle.dump(train_tkn_l, tr_tkn_unk_f)
tr_tkn_unk_f.close()