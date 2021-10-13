import preprocessing as pp


class unigram():
    def __init__(self, tr):
        self.tr = tr
        self.unigramDict = {}
        self.p1 = {}

    def create_uniDict(self):
        for sent in self.tr:
            for i in range(len(sent)):
                if sent[i] in self.unigramDict:
                    self.unigramDict[sent[i]] += 1
                else:
                    self.unigramDict[sent[i]] = 1

    def fit(self):
        self.create_uniDict()
        train_tkn_l = [token for sent in self.tr for token in sent]
        n_token = len(train_tkn_l)
        for wd, count in self.unigramDict.items():
            self.p1[wd] = count/n_token

class bigram(unigram):
    def __init__(self, tr):
        super(bigram, self).__init__(tr)
        self.bigramDict = {}
        #self.biwordList = []
        self.p2 = {}

    def create_uniPDict(self):
        self.create_uniDict()
        train_tkn_l = [token for sent in self.tr for token in sent]
        n_token = len(train_tkn_l)
        for wd, count in self.unigramDict.items():
            self.p1[wd] = count/n_token

    def create_biDict(self):
        self.create_uniPDict()
        # super(bigram, self).create_uniDict()
        for sent in self.tr:
            for i in range(len(sent)-1):
                #self.biwordList.append((self.tr[i], self.tr[i+1]))
                if (sent[i], sent[i+1]) in self.bigramDict:
                    self.bigramDict[(sent[i], sent[i+1])] += 1
                else:
                    self.bigramDict[(sent[i], sent[i+1])]  = 1

    def fit(self):
        self.create_biDict()
        for (wd1, wd2), count in self.bigramDict.items():
            self.p2[(wd1, wd2)] = count/self.unigramDict.get(wd1)


class smoothing1(bigram):
    def __init__(self, tr):
        super(smoothing1, self).__init__(tr)

    def create_uniPDict(self):
        self.create_uniDict()
        train_tkn_l = [token for sent in self.tr for token in sent]
        n_token = len(train_tkn_l)
        V = pp.creat_vocabulary(self.tr)    # create vocabulary
        n_V = len(V)                        # the number of torken types
        for wd, count in self.unigramDict.items():
            self.p1[wd] = (count + 1)/(n_token + n_V)

    def fit(self):
        self.create_biDict()
        V = pp.creat_vocabulary(self.tr)    # create vocabulary
        n_V = len(V)                        # the number of torken types
        for (wd1, wd2), count in self.bigramDict.items():
            self.p2[(wd1, wd2)] = (count+1)/(self.unigramDict.get(wd1)+n_V)



# train_l, test_l = pp.load_data("G://CUNY/NLP/Assignments/HW1/", "train.txt", "test.txt")
# train_tkn_l = pp.token(train_l)
# V = pp.creat_vocabulary(train_tkn_l)
# bi = bigram(train_tkn_l)
# bi.fit()
#
# test = [('hearing', 'your'), ('your', 'reply'), ('reply', '.')]
# for t in test:
#     if t in bi.p2:
#         print('exist')
#     else:
#         print('no')


