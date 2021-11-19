from preprocess2 import process
import pickle
import numpy as np

class NB(process):
    def __init__(self, proc, tr_dir, ts_dir, para_dir, out_dir, train=False):
        self.tr_dir = tr_dir
        self.ts_dir = ts_dir
        self.para_dir = para_dir
        self.out_dir = out_dir          # the output path, including file name
        self.classes = proc.classes
        self.tr_data = proc.tr_data
        self.ts_data = proc.ts_data
        self.v = proc.v
        self.tr_tkn = proc.tr_tkn
        self.ts_tkn = proc.ts_tkn
        if train:
            self.train()

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


    def train(self):
        self.count_p_prior()
        self.count_p_w()


    def pred(self):
        results = []
        for doc in self.ts_tkn['pred']:
            rslt = [','.join(doc)]
            poss = []
            for cl in self.classes:
                pos = 1 * self.p_prior[cl]
                for tkn in doc:
                    if tkn in self.v:
                        pos *= self.p_w[tkn, cl]
                poss.append(pos)
                rslt.append(cl+ ':' +str(pos))
            rslt.append('class:'+ self.classes[np.argmax(poss)])
            rslt = ' '.join(rslt)
            results.append(rslt)
        # write the results into file
        f = open(self.out_dir, 'w')
        for r in results:
            f.write(r+"\n")
        f.close()






proc_f = open('G://CUNY/NLP/Assignments/HW2/movie-review-small.NB', 'rb')
proc = pickle.load(proc_f)
proc_f.close()
nb = NB(proc, 'G://CUNY/NLP/Assignments/HW2/train2.txt', 'G://CUNY/NLP/Assignments/HW2/test2.txt',
        'G://CUNY/NLP/Assignments/HW2/movie-review-small.NB', 'G://CUNY/NLP/Assignments/HW2/small-result.txt', True)
# print(nb.ts_data)
nb.pred()

