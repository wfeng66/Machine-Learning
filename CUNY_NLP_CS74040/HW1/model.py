import preprocessing as pp

class unigram():
    def __init__(self, tr):
        self.tr = tr
        self.unigramDict = {}

    def fit(self):
        for i in range(len(self.tr)):
            if self.tr[i] in self.unigramDict:
                self.unigramDict[self.tr[i]] += 1
            else:
                self.unigramDict[self.tr[i]] = 1

class bigram(unigram):
    def __init__(self, tr):
        super(bigram, self).__init__(tr)
        self.bigramDict = {}
        #self.biwordList = []
        self.p = {}

    def create_biDict(self):
        super(bigram, self).fit()
        for i in range(len(self.tr)-1):
            #self.biwordList.append((self.tr[i], self.tr[i+1]))
            if (self.tr[i], self.tr[i+1]) in self.bigramDict:
                self.bigramDict[(self.tr[i], self.tr[i+1])] += 1
            else:
                self.bigramDict[(self.tr[i], self.tr[i + 1])] = 1

    def fit(self):
        self.create_biDict()
        for (wd1, wd2), count in self.bigramDict.items():
            self.p[(wd1, wd2)] = count/self.unigramDict.get(wd1)


class smoothing1(bigram):
    def __init__(self, tr):
        super(smoothing1, self).__init__(tr)

    def fit(self):
        self.create_biDict()
        V = pp.creat_vocabulary(self.tr)   # create vocabulary
        n_V = len(V)                # the number of torken types
        for (wd1, wd2), count in self.bigramDict.items():
            self.p[(wd1, wd2)] = (count+1)/(self.unigramDict.get(wd1)+n_V)



train_l, test_l = pp.load_data("G://CUNY/NLP/Assignments/HW1/", "train.txt", "test.txt")
train_tkn_l = pp.token(train_l, test_l)
V = pp.creat_vocabulary(train_tkn_l)
bi = bigram(train_tkn_l)
bi.fit()
print(bi.p)


