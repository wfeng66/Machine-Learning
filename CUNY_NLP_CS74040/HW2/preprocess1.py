class process():
    def __init__(self, tr_dir, ts_dir):
        # get rid of the last '/' chart if it has
        if tr_dir[-1] == '/':
            tr_dir = tr_dir[:-1]
        if ts_dir[-1] == '/':
            ts_dir = ts_dir[:-1]
        self.tr_dir = tr_dir                        # training data path
        self.ts_dir = ts_dir                        # test data path
        self.tr_data, self.ts_data = {}, {}         # store training data and test data
        self.read_data()                            # read training and test data
        self.v = self.load_voc()                    # the list of vocabulary
        # stop word set
        self.stw = {'those', 'on', 'own', '’ve', 'yourselves', 'around', 'between', 'four', 'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'regarding', 'hereafter', 'front', 'too', 'used', 'wherein', '‘ll', 'doing', 'everything', 'up', 'onto', 'never', 'either', 'how', 'before', 'anyway', 'since', 'through', 'amount', 'now', 'he', 'was', 'have', 'into', 'because', 'not', 'therefore', 'they', 'n’t', 'even', 'whom', 'it', 'see', 'somewhere', 'thereupon', 'nothing', 'whereas', 'much', 'whenever', 'seem', 'until', 'whereby', 'at', 'also', 'some', 'last', 'than', 'get', 'already', 'our', 'once', 'will', 'noone', "'m", 'that', 'what', 'thus', 'no', 'myself', 'out', 'next', 'whatever', 'although', 'though', 'which', 'would', 'therein', 'nor', 'somehow', 'whereupon', 'besides', 'whoever', 'ourselves', 'few', 'did', 'without', 'third', 'anything', 'twelve', 'against', 'while', 'twenty', 'if', 'however', 'herself', 'when', 'may', 'ours', 'six', 'done', 'seems', 'else', 'call', 'perhaps', 'had', 'nevertheless', 'where', 'otherwise', 'still', 'within', 'its', 'for', 'together', 'elsewhere', 'throughout', 'of', 'others', 'show', '’s', 'anywhere', 'anyhow', 'as', 'are', 'the', 'hence', 'something', 'hereby', 'nowhere', 'latterly', 'say', 'does', 'neither', 'his', 'go', 'forty', 'put', 'their', 'by', 'namely', 'could', 'five', 'unless', 'itself', 'is', 'nine', 'whereafter', 'down', 'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who', 'yourself', 'every', 'thru', 'except', 'very', 'several', 'among', 'being', 'be', 'mine', 'further', 'n‘t', 'here', 'during', 'why', 'with', 'just', "'s", 'becomes', '’ll', 'about', 'a', 'using', 'seeming', "'d", "'ll", "'re", 'due', 'wherever', 'beforehand', 'fifty', 'becoming', 'might', 'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least', 'someone', 'often', 'from', 'keep', 'him', 'or', '‘m', 'top', 'her', 'nobody', 'sometime', 'across', '‘s', '’re', 'hundred', 'only', 'via', 'name', 'eight', 'three', 'back', 'to', 'all', 'became', 'move', 'me', 'we', 'formerly', 'so', 'i', 'whence', 'under', 'always', 'himself', 'in', 'herein', 'more', 'after', 'themselves', 'you', 'above', 'sixty', 'them', 'your', 'made', 'indeed', 'most', 'everywhere', 'fifteen', 'but', 'must', 'along', 'beside', 'hers', 'side', 'former', 'anyone', 'full', 'has', 'yours', 'whose', 'behind', 'please', 'ten', 'seemed', 'sometimes', 'should', 'over', 'take', 'each', 'same', 'rather', 'really', 'latter', 'and', 'ca', 'hereupon', 'part', 'per', 'eleven', 'ever', '‘re', 'enough', "n't", 'again', '‘d', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile', 'whither', 'there', 'toward', '’m', "'ve", '’d', 'give', 'do', 'an', 'quite', 'these', 'everyone', 'towards', 'this', 'cannot', 'afterwards', 'beyond', 'make', 'were', 'whether', 'well', 'another', 'below', 'first', 'upon', 'any', 'none', 'many', 'serious', 'various', 're', 'two', 'less', '‘ve', 'br'}


    def read_data(self):
        '''
        load the training data and test data
        no parameter
        :return:    no return, all results are store as properties of the object
        '''
        import os
        self.classes = [name for name in os.listdir(self.tr_dir)        # retrieve classes
                    if os.path.isdir(os.path.join(self.tr_dir, name))]
        for cat in self.classes:                                        # iterate classes and read the files
            self.tr_data[cat] = self.read_OneClass(self.tr_dir, cat)
            self.ts_data[cat] = self.read_OneClass(self.ts_dir, cat)


    def read_OneClass(self, path,cat):
        '''
        load given class files
        :param path:    (string) the parent directory of the files
        :param cat:     (string) the subdirectory, which should have the same name as the class label
        :return:        (2D list) each line one document
        '''
        doc_lst = []            # use to store all document in given class
        import os
        file_lst = os.listdir(path+ '/' + cat )
        for file in file_lst:   # iterate the files
            doc = []            # use to store individual document
            # read the file
            with open(path + '/' + cat + '/' +file, 'r', encoding='UTF-8') as f:
                for line in f:
                    doc.append(line)
            doc_lst.append(doc)
        return doc_lst


    def load_voc(self):
        '''
        load the vocabulary from file
        no parameter
        :return:    (list) vocabulary
        '''
        import os
        # create the path for vocabulary file
        path = os.path.abspath(os.path.join(self.tr_dir, os.pardir))
        voc = []
        # read the vocabulary file
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
        lowercase and tokenize document, geting rid of the punctuations
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

    def rmvDigitTags(self, doc):
        '''
        remove the digits and some html tags in document
        combine the documents into one document before calling this function
        :param doc: (list) the document to be removed the digits, each element is one document
        :return:    (list) the cleaned document
        '''
        # from string import digits
        import re
        result = [re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", d)
                  for d in doc]
        result = [re.sub(r"\d+", "", d) for d in result]
        return result


    def expandVoc(self):
        '''
        expand the vocabulary with training data set
        no parameter
        :return:    (set) new vocabulary which union original vocabulary and word types encountered in training data set
        '''
        result = set()
        for cl in self.classes:                                     # iterate classes
            result = set.union(result, set(self.tr_tkn[cl]))
        return set.union(set(self.v), result)


    def rmvSTW(self, lst):
        '''
        remove the stop words from inputted list
        :param lst: (list) list of features, tokens
        :return:    (list) list of features without stop of word
        '''
        return [tkn for tkn in lst if tkn not in self.stw]


    def fit(self):
        '''
        the main preprocess
        no parameter
        :return:    no return, all results are saved as the properties of the object
        '''
        self.tr_tkn, self.ts_tkn = {}, {}
        for cl in self.classes:
            # clean training data
            self.tr_tkn[cl] = self.token(self.rmvDigitTags(self.combDoc(self.tr_data[cl])))
            self.tr_tkn[cl] = self.rmvSTW(self.tr_tkn[cl])
            # clean test data
            self.ts_data[cl] = list(map(self.rmvDigitTags, self.ts_data[cl]))
            self.ts_tkn[cl] = list(map(self.token, self.ts_data[cl]))
            self.ts_tkn[cl] = list(map(self.rmvSTW, self.ts_tkn[cl]))

        # expand vocabulary
        self.v = self.expandVoc()
        self.v = set(self.rmvSTW(list(self.v)))


def main(tr_dir, ts_dir, para_dir):
    import pickle
    proc = process(tr_dir, ts_dir)      # instantiate the process
    proc.fit()                          # preprocess
    # write to file
    proc_f= open(para_dir, 'wb')
    pickle.dump(proc, proc_f)
    proc_f.close()


if __name__ == '__main__':
    # arguments parse
    import argparse
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('tr_dir', type=str, help='training path')
    parser.add_argument('ts_dir', type=str, help='test path')
    parser.add_argument('para_dir', type=str, help='parameter path')
    args = parser.parse_args()
    # call the main function
    main(tr_dir = args.tr_dir, ts_dir = args.ts_dir, para_dir = args.para_dir)



