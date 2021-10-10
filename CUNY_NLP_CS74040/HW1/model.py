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
    





train_l, test_l = pp.load_data("G://CUNY/NLP/Assignments/HW1/", "train.txt", "test.txt")
train_tkn_l = pp.token(train_l, test_l)
V = pp.creat_vocabulary(train_tkn_l)
uni = unigram(train_tkn_l)
uni.fit()
print(uni.unigramDict)


