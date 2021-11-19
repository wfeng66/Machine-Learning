class process():
    def __init__(self, tr_dir, ts_dir):
        if tr_dir[-1] == '/':
            tr_dir = tr_dir[:-1]
        if ts_dir[-1] == '/':
            ts_dir = ts_dir[:-1]
        self.tr_dir = tr_dir
        self.ts_dir = ts_dir
        self.read_data
        self.tr_data, self.ts_data = {}, {}
        self.read_data()                # read training and test data
        self.v = self.load_voc()             # the list of vocabulary


    def read_data(self):
        import os
        self.classes = [name for name in os.listdir(self.tr_dir)        # retrieve classes
                    if os.path.isdir(os.path.join(self.tr_dir, name))]
        for cat in self.classes:
            self.tr_data[cat] = self.read_OneClass(self.tr_dir, cat)
            self.ts_data[cat] = self.read_OneClass(self.ts_dir, cat)


    def read_OneClass(self, path,cat):
        doc_lst = []            # use to store all document in given class
        import os
        file_lst = os.listdir(path+ '/' + cat )
        for file in file_lst:
            doc = []            # use to store individual document
            with open(path + '/' + cat + '/' +file, 'r', encoding='UTF-8') as f:
                for line in f:
                    doc.append(line)
            doc_lst.append(doc)
        return doc_lst

    def load_voc(self):
        import os
        path = os.path.abspath(os.path.join(self.tr_dir, os.pardir))
        voc = []
        with open(os.path.join(path, 'imdb.vocab')) as f:
            for line in f:
                voc.append(line.strip())
        return voc

    def combDoc(self, docs):
        '''
        Combine the documents in identical class into one document
        :parameter: docs(2d list) - documents to be combined
        :return:    (1d list) combined document
        '''
        combDoc = []
        for doc in docs:
            combDoc = combDoc + doc
        return combDoc


    def token(self, doc):
        '''
        lowercase and tokenize document
        :param doc: 1d list, the document to be tokenized
        :return:    1d list, tokens list for doc
        '''
        words = list()
        from tqdm import tqdm
        # split the sentences to tokens
        for d in tqdm(doc):
            words = words + d.lower().split()
        # get rid of punctuations
        import string
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table)  for w in tqdm(words)]
        return stripped

    def count_p_prior(self):
        '''

        :return:
        '''
        self.p_prior = {}
        total_doc = 0
        for cl in self.classes:
            self.p_prior[cl] = len(self.tr_data[cl])
            total_doc += self.p_prior[cl]
        for cl in self.p_prior:
            self.p_prior[cl] = self.p_prior[cl]/total_doc

    def count_p_w(self):
        '''

        :return:
        '''
        from tqdm import tqdm
        num_v = len(self.v)
        self.p_w = {}           # the likelihood with add-1 smooth
        num_tkn = {}            # total token in all documents given class
        for cl in self.classes:
            num_tkn[cl] = len(self.tr_tkn[cl])
            for w in tqdm(self.v):
                self.p_w[(w, cl)] = (self.tr_tkn[cl].count(w) + 1)/(num_tkn[cl] + num_v)


    def fit(self):
        self.tr_tkn = {}
        for cl in self.classes:
            self.tr_tkn[cl] = self.token(self.combDoc(self.tr_data[cl]))
        self.count_p_w()
        self.count_p_w()
        # import pickle
        # model_f= open( path + 'model.txt', 'wb')
        # pickle.dump()


    def test(self):
        self.count_p_prior()
        self.fit()
        self.count_p_w()
        print(self.p_w)
        print(len(self.p_w))
        print(len(self.v))



#
# nb = NB('G://CUNY/NLP/Assignments/HW2/train', 'G://CUNY/NLP/Assignments/HW2/test')
# nb.test()
