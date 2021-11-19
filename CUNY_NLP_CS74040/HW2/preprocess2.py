class process():
    def __init__(self, tr_dir, ts_dir):
        '''

        :param tr_dir: string, the absolute path of training data set, including file name
        :param ts_dir: string, the absolute path of test data set, including file name
        :param para_dir: string, the absolute path of parameters file, including file name
        '''
        self.tr_dir = tr_dir
        self.ts_dir = ts_dir
        # self.para_dir = tr_dir
        self.tr_data, self.ts_data = {}, {}
        self.read_data()                # read training and test data
        # self.v = self.load_voc()             # the list of vocabulary
        self.fit()


    def read_data(self):
        '''

        :return:
        '''
        classes = set()
        # read the training file
        file_tr = open(self.tr_dir, 'r')
        lines_tr = file_tr.readlines()
        # iterate examples and store to self.tr_data
        # it requires the first column in training data file is the label and each line for one example
        for line_tr in lines_tr:
            example_tr = line_tr.split()
            classes.add(example_tr[0])
            if example_tr[0] not in self.tr_data:
                self.tr_data[example_tr[0]] = []
            self.tr_data[example_tr[0]].append(example_tr[1:])
        self.classes = list(classes)
        # read the test file
        file_ts = open(self.ts_dir, 'r')
        lines_ts = file_ts.readlines()
        ts_data = []
        # iterate the examples in test file
        # no label in the test file
        for line_ts in lines_ts:
            example_ts = line_ts.split()
            ts_data.append(example_ts)
        self.ts_data['pred'] = ts_data      # the key of 'pred' indicate this test data doesn't include label



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


    def create_v(self):
        '''
        create vocabulary
        :return: no, the result save in self.v
        '''
        self.v = set()
        for cl in self.classes:
            self.v.update(set(self.tr_tkn[cl]))



    def fit(self):
        self.tr_tkn, self.ts_tkn = {}, {}
        for cl in self.classes:
            self.tr_tkn[cl] = self.token(self.combDoc(self.tr_data[cl]))
        self.create_v()
        self.ts_tkn['pred'] = list(map(self.token, self.ts_data['pred']))
        # self.count_p_w()
        # self.count_p_w()
        # import pickle
        # model_f= open( path + 'model.txt', 'wb')
        # pickle.dump()


    def test(self):
        self.fit()
        print(self.tr_data)
        print(self.ts_data)
        print(self.classes)
        print(len(self.v))
        print(self.v)
        # self.count_p_prior()
        # self.fit()
        # self.count_p_w()
        # print(self.p_w)
        # print(len(self.p_w))
        # print(len(self.v))



def main(tr_dir, ts_dir, para_dir):
    import pickle
    proc = process(tr_dir, ts_dir)
    proc_f= open(para_dir, 'wb')
    pickle.dump(proc, proc_f)
    proc_f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('tr_dir', type=str, help='training path')
    parser.add_argument('ts_dir', type=str, help='test path')
    parser.add_argument('para_dir', type=str, help='parameter path')
    args = parser.parse_args()
    main(tr_dir = args.tr_dir, ts_dir = args.ts_dir, para_dir = args.para_dir)

#
# nb = process('G://CUNY/NLP/Assignments/HW2/train2.txt', 'G://CUNY/NLP/Assignments/HW2/test2.txt')
# nb.test()
# run
# python preprocess2.py G://CUNY/NLP/Assignments/HW2/train2.txt G://CUNY/NLP/Assignments/HW2/test2.txt G://CUNY/NLP/Assignments/HW2/movie-review-small.NB
