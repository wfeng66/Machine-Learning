import preprocessing as pp
from tqdm import tqdm

class unigram():
    def __init__(self, tr):
        # constructor need the training corpus as input
        # the training corpus, tr should be 2d tokenized list, each row corresponing to a sentence
        # the tr should include punctuations, padding, lower-case, <unk> token,
        # any preprocessing you need should be done before feed into the constructor
        self.tr = tr                        # training corpus
        self.unigramDict = {}               # unigram dictionary
        self.p1 = {}                        # store the unigram probabilities

    def create_uniDict(self):
        # this function used for counting the tokens
        for sent in self.tr:                # iterate sentences
            for i in range(len(sent)):      # iterate tokens
                if sent[i] in self.unigramDict:         # the token exist in the dictionary
                    self.unigramDict[sent[i]] += 1
                else:                                   # never seen the token in the dictionary
                    self.unigramDict[sent[i]] = 1

    def fit(self):
        # train the model
        # calculate the probabilities for each token
        self.create_uniDict()
        # count the total number of tokens in the training corpus
        train_tkn_l = [token for sent in self.tr for token in sent]
        n_token = len(train_tkn_l)
        # create the probabilities dictionary
        for wd, count in self.unigramDict.items():
            self.p1[wd] = count/n_token

class bigram(unigram):
    def __init__(self, tr):
        # constructor need the training corpus as input
        # the training corpus, tr should be 2d tokenized list, each row corresponing to a sentence
        # the tr should include punctuations, padding, lower-case, <unk> token,
        # any preprocessing you need should be done before feed into the constructor
        super(bigram, self).__init__(tr)
        self.bigramDict = {}                # bigram dictionary
        self.p2 = {}                        # store the bigram probabilities

    def create_uniPDict(self):
        # same as the fit() method in unigram, which create the unigram probabilities for each token
        self.create_uniDict()
        train_tkn_l = [token for sent in self.tr for token in sent]
        n_token = len(train_tkn_l)
        for wd, count in self.unigramDict.items():
            self.p1[wd] = count/n_token

    def create_biDict(self):
        # count the bigram for each work-pairs
        self.create_uniPDict()
        for sent in self.tr:                # iterate sentences
            for i in range(len(sent)-1):    # iterate tokens
                if (sent[i], sent[i+1]) in self.bigramDict:     # the token pair exist in bigramDict
                    self.bigramDict[(sent[i], sent[i+1])] += 1
                else:                                           # new token pair
                    self.bigramDict[(sent[i], sent[i+1])]  = 1

    def fit(self):
        # train the model
        # calculate the probabilities for each token pair
        self.create_biDict()
        for (wd1, wd2), count in self.bigramDict.items():
            self.p2[(wd1, wd2)] = count/self.unigramDict.get(wd1)


class smoothing1(bigram):
    def __init__(self, tr):
        # constructor need the training corpus as input
        # the training corpus, tr should be 2d tokenized list, each row corresponing to a sentence
        # the tr should include punctuations, padding, lower-case, <unk> token,
        # any preprocessing you need should be done before feed into the constructor
        super(smoothing1, self).__init__(tr)

    def create_uniPDict(self):
        # count the unigram for each token
        self.create_uniDict()
        # count the total number of the token
        train_tkn_l = [token for sent in self.tr for token in sent]
        n_token = len(train_tkn_l)
        # count the total number of vocabulary, word types
        V = pp.creat_vocabulary(self.tr)    # create vocabulary
        n_V = len(V)                        # the number of torken types
        # count the unigram with add-1 smoothing
        for wd, count in self.unigramDict.items():
            self.p1[wd] = (count + 1)/(n_token + n_V)

    def fit(self):
        # train the model
        # creat bigram with add-1 smoothing for each token pair
        self.create_biDict()
        V = pp.creat_vocabulary(self.tr)    # create vocabulary
        n_V = len(V)                        # the number of torken types
        for (wd1, wd2), count in self.bigramDict.items():   # iterate token pairs in bigramDict
            self.p2[(wd1, wd2)] = (count+1)/(self.unigramDict.get(wd1)+n_V)


class katz(bigram):
    def __init__(self, tr, c=0.5):          # c is the discount constant
        # constructor need the training corpus as input
        # the training corpus, tr should be 2d tokenized list, each row corresponing to a sentence
        # the tr should include punctuations, padding, lower-case, <unk> token,
        # any preprocessing you need should be done before feed into the constructor
        super(katz, self).__init__(tr)
        self.bigramDict_disc = {}       # store the discount counts for bigram
        self.a = {}                     # store leftover probabilities
        self.c = c                      # the discount constant

    def fit(self):
        # train the model
        print('Training katz model......')
        self.create_biDict()
        # create discounted counts for bigram pairs
        self.bigramDict_disc = self.bigramDict.copy()
        for (wd1, wd2), count in self.bigramDict_disc.items():
            self.bigramDict_disc[(wd1, wd2)] = count - self.c
        # create discounted probabilities p*
        for (wd1, wd2), count_disc in tqdm(self.bigramDict_disc.items()):
            self.p2[(wd1, wd2)] = count_disc/self.unigramDict.get(wd1)
            # create leftover probabilities a for each wd1
            wd1_cnt_sum = sum([c for (w1, w2), c in self.bigramDict_disc.items() if w1 == wd1])
            self.a[wd1] = 1-wd1_cnt_sum/self.unigramDict.get(wd1)





