from preprocess2 import process
import pickle
import numpy as np
import math as m

class NB(process):
    def __init__(self, proc, out_dir, train=False):
        self.out_dir = out_dir          # the output path, including file name
        self.classes = proc.classes     # the classes in this model
        self.tr_data = proc.tr_data     # store training data
        self.ts_data = proc.ts_data     # store test data
        self.v = proc.v                 # store vocabulary
        self.tr_tkn = proc.tr_tkn       # store training features, tokens
        self.ts_tkn = proc.ts_tkn       # store test features, tokens
        self.train()


    def count_p_prior(self):
        '''
        calculate the prior probabilities for each class
        :return:    no return, the results are stored in the self.p_prior[cl]
        '''
        self.p_prior = {}
        total_doc = 0
        for cl in self.classes:                         # iterate the classes
            self.p_prior[cl] = len(self.tr_data[cl])    # total number of documents in specific class
            total_doc += self.p_prior[cl]               # count the total number of all documents in the data set
        for cl in self.p_prior:
            self.p_prior[cl] = m.log(self.p_prior[cl]/total_doc)    # calculate the prior probabilities for each class in log space


    def count_p_w(self):
        '''
        calculate the likelihood for features
        :return:    no return, the results are stored in the self.p_w[w, cl]
        '''
        from tqdm import tqdm
        num_v = len(self.v)     # total number of features, tokens
        self.p_w = {}           # the likelihood with add-1 smooth
        num_tkn = {}            # total token in all documents given class
        for cl in self.classes:                     # iterate classes
            num_tkn[cl] = len(self.tr_tkn[cl])      # total number of tokens in given class
            for w in tqdm(self.v):                  # iterate the token in vocabulary
                # calculate the likelihood for features in log space
                self.p_w[(w, cl)] = m.log((self.tr_tkn[cl].count(w) + 1)/(num_tkn[cl] + num_v))


    def train(self):
        '''
        train the model, calculate the prior probabilities and likelihood
        :return:    no return, the results are stored in the properties of the class
        '''
        self.count_p_prior()        # calculate the prior probabilities for each class
        self.count_p_w()            # calculate the likelihood for features


    def pred(self):
        '''
        this function is used for inference, no validation
        no parameter
        :return:    no return, the results will be displayed on the monitor and output to file
                    whose path and file name are indicated in self.out_dir
        '''
        results = []                        # list for storing output information
        for doc in self.ts_tkn['pred']:     # iterate documents to be predicted
            rslt = self.docPred(doc)        # inference
            rslt = ' '.join(rslt)
            results.append(rslt)
        # write the results into file
        f = open(self.out_dir, 'w')
        for r in results:
            f.write(r+"\n")
        f.close()


    def validate(self):
        '''
        validation
        no parameter
        :return:    no return, the final results will be displayed on the monitor and output to files
                    whose path and file name are indicated in self.out_dir
        '''
        results, self.miscl = [], []        # self.miscl is used to store misclassified documents (examples)
        num_corr = 0                        # the number of correctly classified documents
        total = 0                           # initialize the total, which store the total number of documents in test data set
        for label in self.classes:          # iterate classes
            total += len(self.ts_tkn[label])        # plus the number of documents in specific class
            for doc in self.ts_tkn[label]:          # iterate documents in specific class
                rslt = self.docPred(doc)            # inference
                if rslt[-1] == label:               # if the prediction is correct
                    num_corr = num_corr + 1
                else:                               # if misclassified
                    self.miscl.append([rslt, rslt[-1], label])
                rslt = [rslt[0], rslt[-1], label]
                results.append(rslt)
        print('total: ', total)
        self.accuracy = float(num_corr)/float(total)        # accuracy
        print('acc: ', self.accuracy)
        # write the results into file
        f = open(self.out_dir, 'w', encoding='utf-8')
        for r in results:
            f.write(' '.join(r)+"\n")
        f.write('overall accuracy: ' + str(self.accuracy))
        f.close()
        # write to pandas dataframe for analysis
        import pandas as pd
        results_df = pd.DataFrame(results, columns=['review', 'predicted', 'label'])
        results_df.to_csv('G://CUNY/NLP/Assignments/HW2/big-result.csv')
        err_df = results_df[results_df.predicted != results_df.label]
        err_df.to_csv('G://CUNY/NLP/Assignments/HW2/big-error.csv')


    def docPred(self, doc):
        '''
        inference single document (example)
        :param doc:     (list) list of token in single document
        :return:        (list) the result for output
                        first column is a string including all the tokens in the document
                        the possibilities in each class
                        the final predicted class
        '''
        rslt = [','.join(doc)]      # initialize the result list to include the tokens in the document to be predict
        poss = []                   # store possibilities for each class
        for cl in self.classes:     # iterate classes
            pos = self.p_prior[cl]  # initialize possibility by prior probability of the class
            for tkn in doc:         # iterate tokens in the document
                if tkn in self.v:   # if seen in training data
                    pos += self.p_w[tkn, cl]
            print('total posibility-'+cl+': ', pos, sep=' ')
            poss.append(pos)
            rslt.append(cl + ':' + str(pos))
        print('result: ', self.classes[np.argmax(poss)])
        rslt.append(self.classes[np.argmax(poss)])
        return rslt


def main(proc_dir, para_dir, out_dir, train=False, pred = False):
    '''
    the main process, training or laoding the model, predict or validate the test data
    :param proc_dir:    (string) the path of preprocess output file, including file name
    :param para_dir:    (string) the path of parameters file, including file name
    :param out_dir:     (string) the path of validation result file, including file name
    :param train:       (boolean) if True, train the model; if False, load the trained parameters
    :param pred:        (boolean) predict or validate.
    :return:            (Noon) all the results are output to files
    '''
    import pickle
    print(train)

    if train == 'True' or train == 'true':                        # train the model
        print('training...')
        # load the preprocess results from dump file
        proc_f = open(proc_dir, 'rb')
        proc = pickle.load(proc_f)
        proc_f.close()
        nb = NB(proc, out_dir, train)        # instantiate NB object
        # write the parameter file for NB oject
        nb_f = open(para_dir, 'wb')
        pickle.dump(nb, nb_f)
        nb_f.close()
    else:                                   # instead of training, load the model
        print('loading...')
        nb_f = open(para_dir, 'rb')
        nb = pickle.load(nb_f)
        nb_f.close()
        print('pred: ', pred)
    if pred == 'True' or pred == 'true':                        # predict - inference only
        print('inferencing...')
        nb.pred()
    else:                                   # validate - using true label verify and output precision
        print('validating...')
        nb.validate()



if __name__ == '__main__':
    # arguments parse
    import argparse
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('proc_dir', type=str, help='preprocessed file')
    parser.add_argument('para_dir', type=str, help='parameter file')
    parser.add_argument('out_dir', type=str, help='output file')
    parser.add_argument('train', type=str, help='training reaquired')
    parser.add_argument('pred', type=str, help='predict or validate')
    args = parser.parse_args()
    # call main function
    main(proc_dir = args.proc_dir, para_dir = args.para_dir, out_dir = args.out_dir, train = args.train, pred = args.pred)


# python NB.py G://CUNY/NLP/Assignments/HW2/movie-review-small.NB Noon Noon True True
# python NB.py G://CUNY/NLP/Assignments/HW2/movie_review_big.NB G://CUNY/NLP/Assignments/HW2/movie-review-BOW.NB G://CUNY/NLP/Assignments/HW2/big-result.txt True False


# load small data set parameter
# proc_f = open('G://CUNY/NLP/Assignments/HW2/movie_review_big.NB', 'rb')
# proc = pickle.load(proc_f)
# proc_f.close()
# nb = NB(proc, 'G://CUNY/NLP/Assignments/HW2/train', 'G://CUNY/NLP/Assignments/HW2/test',
#        'G://CUNY/NLP/Assignments/HW2/movie_review_big.NB', 'G://CUNY/NLP/Assignments/HW2/big-result.txt', True)
# # print(nb.ts_data)
# write parameter file
# nb_f = open('G://CUNY/NLP/Assignments/HW2/movie-review-BOW.NB', 'wb')
# pickle.dump(nb, nb_f)
# nb_f.close()

# nb_f = open('G://CUNY/NLP/Assignments/HW2/movie-review-BOW.NB', 'rb')
# nb = pickle.load(nb_f)
# nb_f.close()
# nb.validate()
# print(nb.miscl)
# nb.test()

# for tkn in proc.ts_tkn['pos'][0]:
#     if tkn == 'this':
#         proc.ts_tkn['pos'][0].remove(tkn)
#
# print(proc.ts_tkn['pos'][0])




